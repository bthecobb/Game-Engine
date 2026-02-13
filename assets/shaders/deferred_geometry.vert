#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aColor;      // Vertex color (for procedural buildings)
layout (location = 4) in vec3 aEmissive;   // Emissive color (glowing windows)
layout (location = 5) in vec3 aTangent;    // Tangent (for normal mapping, optional)

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 VertexColor;
out vec3 Emissive;
out vec3 Tangent;
out vec3 Bitangent;
out vec4 FragPosLightSpace;
// For motion vectors (current and previous clip-space positions)
out vec4 CurrClipPos;
out vec4 PrevClipPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;
uniform mat3 normalMatrix;
// Previous-frame transforms for motion vector calculation
uniform mat4 prevModel;
uniform mat4 prevView;
uniform mat4 prevProjection;

void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    FragPosLightSpace = lightSpaceMatrix * worldPos;

    Normal = normalize(normalMatrix * aNormal);
    TexCoord = aTexCoord;
    VertexColor = aColor;
    Emissive = aEmissive;

    // Tangent space (only if tangent is provided)
    if (length(aTangent) > 0.01) {
        Tangent = normalize(normalMatrix * aTangent);
        Bitangent = normalize(normalMatrix * cross(aNormal, aTangent));
    } else {
        Tangent = vec3(1, 0, 0);
        Bitangent = vec3(0, 1, 0);
    }

    // Compute current and previous clip positions
    CurrClipPos = projection * view * worldPos;
    vec4 prevWorldPos = prevModel * vec4(aPos, 1.0);
    PrevClipPos = prevProjection * prevView * prevWorldPos;

    gl_Position = CurrClipPos;
}
