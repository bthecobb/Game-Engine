#include "Rendering/Skybox.h"
#include "Rendering/SkyboxUtils.h"
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

// Use stb_image for HDR loading (implementation already defined in Mesh.cpp)
#include "stb_image.h"

namespace CudaGame {
namespace Rendering {

// Skybox vertex shader source (inline for simplicity)
static const char* skyboxVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 WorldPos;

uniform mat4 projection;
uniform mat4 view;
uniform float rotation;

void main()
{
    // Apply rotation around Y axis
    float c = cos(rotation);
    float s = sin(rotation);
    mat3 rotMat = mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
    
    WorldPos = rotMat * aPos;
    
    // Remove translation from view matrix
    mat4 viewNoTranslation = mat4(mat3(view));
    gl_Position = projection * viewNoTranslation * vec4(WorldPos, 1.0);
    gl_Position = gl_Position.xyww; // Ensure depth = 1.0 after perspective divide
}
)";

// Skybox fragment shader with tone mapping
static const char* skyboxFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform samplerCube environmentMap;
uniform float exposure;
uniform float gamma;

void main()
{
    vec3 envColor = texture(environmentMap, WorldPos).rgb;
    
    // Tone mapping (Reinhard)
    vec3 mapped = vec3(1.0) - exp(-envColor * exposure);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    
    FragColor = vec4(mapped, 1.0);
}
)";

// Equirectangular to cubemap conversion shader (vertex)
static const char* equirectVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 WorldPos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    WorldPos = aPos;
    gl_Position = projection * view * vec4(WorldPos, 1.0);
}
)";

// Equirectangular to cubemap conversion shader (fragment)
static const char* equirectFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform sampler2D equirectangularMap;

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

void main()
{
    vec2 uv = SampleSphericalMap(normalize(WorldPos));
    vec3 color = texture(equirectangularMap, uv).rgb;
    
    FragColor = vec4(color, 1.0);
}
)";

Skybox::Skybox() {}

Skybox::~Skybox() {
    Shutdown();
}

bool Skybox::LoadHDR(const std::string& hdrPath, int cubemapSize) {
    std::cout << "[Skybox] Loading HDR environment: " << hdrPath << std::endl;
    
    // Initialize shaders if not already done
    if (!m_initialized) {
        if (!InitializeShaders()) {
            std::cerr << "[Skybox] Failed to initialize shaders" << std::endl;
            return false;
        }
        CreateCubeGeometry();
        m_initialized = true;
    }
    
    // Load HDR image
    float* hdrData = nullptr;
    int width = 0, height = 0;
    if (!LoadHDRImage(hdrPath, hdrData, width, height)) {
        return false;
    }
    
    // Bake to cubemap
    bool success = BakeToCubemap(hdrData, width, height, cubemapSize);
    
    // Free HDR data
    stbi_image_free(hdrData);
    
    if (success) {
        std::cout << "[Skybox] Successfully loaded and baked HDR to " << cubemapSize << "x" << cubemapSize << " cubemap" << std::endl;
    }
    
    return success;
}

bool Skybox::LoadHDRImage(const std::string& path, float*& data, int& width, int& height) {
    stbi_set_flip_vertically_on_load(true);
    int nrComponents;
    data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);
    
    if (!data) {
        std::cerr << "[Skybox] Failed to load HDR image: " << path << std::endl;
        std::cerr << "[Skybox] STB Error: " << stbi_failure_reason() << std::endl;
        return false;
    }
    
    std::cout << "[Skybox] Loaded HDR image: " << width << "x" << height << ", " << nrComponents << " components" << std::endl;
    return true;
}

bool Skybox::BakeToCubemap(const float* hdrData, int hdrWidth, int hdrHeight, int cubemapSize) {
    // Create equirectangular texture
    GLuint hdrTexture;
    glGenTextures(1, &hdrTexture);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, hdrWidth, hdrHeight, 0, GL_RGB, GL_FLOAT, hdrData);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Create cubemap texture
    if (m_cubemapTexture == 0) {
        glGenTextures(1, &m_cubemapTexture);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubemapTexture);
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, cubemapSize, cubemapSize, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Create FBO for rendering to cubemap faces
    if (m_captureFBO == 0) {
        glGenFramebuffers(1, &m_captureFBO);
        glGenRenderbuffers(1, &m_captureRBO);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, cubemapSize, cubemapSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_captureRBO);
    
    // Set up projection and view matrices for the 6 cubemap faces
    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] = {
        glm::lookAt(glm::vec3(0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };
    
    // Use equirect shader and render to each cubemap face
    glUseProgram(m_equirectShader);
    glUniform1i(glGetUniformLocation(m_equirectShader, "equirectangularMap"), 0);
    glUniformMatrix4fv(glGetUniformLocation(m_equirectShader, "projection"), 1, GL_FALSE, glm::value_ptr(captureProjection));
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);
    
    glViewport(0, 0, cubemapSize, cubemapSize);
    glBindFramebuffer(GL_FRAMEBUFFER, m_captureFBO);
    
    for (unsigned int i = 0; i < 6; ++i) {
        glUniformMatrix4fv(glGetUniformLocation(m_equirectShader, "view"), 1, GL_FALSE, glm::value_ptr(captureViews[i]));
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, m_cubemapTexture, 0);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        RenderCube();
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Cleanup temporary texture
    glDeleteTextures(1, &hdrTexture);
    
    return true;
}

void Skybox::RenderCube() {
    if (m_cubeVAO == 0) return;
    glBindVertexArray(m_cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void Skybox::Render(const glm::mat4& view, const glm::mat4& projection) {
    if (!m_enabled || m_cubemapTexture == 0 || m_shaderProgram == 0) {
        return;
    }
    
    // Set depth function to less than or equal for skybox rendering
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE); // Don't write to depth buffer
    
    glUseProgram(m_shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1f(glGetUniformLocation(m_shaderProgram, "rotation"), m_rotation);
    glUniform1f(glGetUniformLocation(m_shaderProgram, "exposure"), m_exposure);
    glUniform1f(glGetUniformLocation(m_shaderProgram, "gamma"), m_gamma);
    glUniform1i(glGetUniformLocation(m_shaderProgram, "environmentMap"), 0);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubemapTexture);
    
    RenderCube();
    
    // Restore default depth state
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

void Skybox::Shutdown() {
    if (m_cubemapTexture) {
        glDeleteTextures(1, &m_cubemapTexture);
        m_cubemapTexture = 0;
    }
    
    if (m_shaderProgram) {
        glDeleteProgram(m_shaderProgram);
        m_shaderProgram = 0;
    }
    
    if (m_equirectShader) {
        glDeleteProgram(m_equirectShader);
        m_equirectShader = 0;
    }
    
    if (m_captureFBO) {
        glDeleteFramebuffers(1, &m_captureFBO);
        m_captureFBO = 0;
    }
    
    if (m_captureRBO) {
        glDeleteRenderbuffers(1, &m_captureRBO);
        m_captureRBO = 0;
    }
    
    DestroyCubeGeometry();
    m_initialized = false;
}

bool Skybox::InitializeShaders() {
    // Compile and link skybox shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &skyboxVertexShader, NULL);
    glCompileShader(vertexShader);
    
    // Check for vertex shader compilation errors
    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "[Skybox] Vertex shader compilation failed:\n" << infoLog << std::endl;
        return false;
    }
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &skyboxFragmentShader, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "[Skybox] Fragment shader compilation failed:\n" << infoLog << std::endl;
        glDeleteShader(vertexShader);
        return false;
    }
    
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_shaderProgram, 512, NULL, infoLog);
        std::cerr << "[Skybox] Shader program linking failed:\n" << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Compile and link equirect-to-cubemap shader
    GLuint equirectVertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(equirectVertex, 1, &equirectVertexShader, NULL);
    glCompileShader(equirectVertex);
    
    glGetShaderiv(equirectVertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(equirectVertex, 512, NULL, infoLog);
        std::cerr << "[Skybox] Equirect vertex shader compilation failed:\n" << infoLog << std::endl;
        return false;
    }
    
    GLuint equirectFragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(equirectFragment, 1, &equirectFragmentShader, NULL);
    glCompileShader(equirectFragment);
    
    glGetShaderiv(equirectFragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(equirectFragment, 512, NULL, infoLog);
        std::cerr << "[Skybox] Equirect fragment shader compilation failed:\n" << infoLog << std::endl;
        glDeleteShader(equirectVertex);
        return false;
    }
    
    m_equirectShader = glCreateProgram();
    glAttachShader(m_equirectShader, equirectVertex);
    glAttachShader(m_equirectShader, equirectFragment);
    glLinkProgram(m_equirectShader);
    
    glGetProgramiv(m_equirectShader, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_equirectShader, 512, NULL, infoLog);
        std::cerr << "[Skybox] Equirect shader program linking failed:\n" << infoLog << std::endl;
        glDeleteShader(equirectVertex);
        glDeleteShader(equirectFragment);
        return false;
    }
    
    glDeleteShader(equirectVertex);
    glDeleteShader(equirectFragment);
    
    std::cout << "[Skybox] Shaders compiled and linked successfully" << std::endl;
    return true;
}

void Skybox::CreateCubeGeometry() {
    // Unit cube vertices (for rendering skybox)
    float vertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    
    glGenVertexArrays(1, &m_cubeVAO);
    glGenBuffers(1, &m_cubeVBO);
    glBindVertexArray(m_cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

void Skybox::DestroyCubeGeometry() {
    if (m_cubeVAO) {
        glDeleteVertexArrays(1, &m_cubeVAO);
        m_cubeVAO = 0;
    }
    if (m_cubeVBO) {
        glDeleteBuffers(1, &m_cubeVBO);
        m_cubeVBO = 0;
    }
}

} // namespace Rendering
} // namespace CudaGame
