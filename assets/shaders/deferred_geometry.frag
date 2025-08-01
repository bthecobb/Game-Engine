#version 330 core

layout (location = 0) out vec3 gPosition;      // World position
layout (location = 1) out vec3 gNormal;        // World normal
layout (location = 2) out vec4 gAlbedoSpec;    // Albedo + specular
layout (location = 3) out vec3 gMetallicRoughness; // Metallic + Roughness + AO

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 Tangent;
in vec3 Bitangent;
in vec4 FragPosLightSpace;

// PBR Material uniforms
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(normalMap, TexCoord).xyz * 2.0 - 1.0;

    vec3 N = normalize(Normal);
    vec3 T = normalize(Tangent);
    vec3 B = normalize(Bitangent);
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

void main()
{
    gPosition = FragPos;
    
    // Normal mapping
    if (textureSize(normalMap, 0).x > 1) {
        gNormal = getNormalFromMap();
    } else {
        gNormal = normalize(Normal);
    }
    
    // Albedo
    vec3 albedoColor = albedo;
    if (textureSize(albedoMap, 0).x > 1) {
        albedoColor *= texture(albedoMap, TexCoord).rgb;
    }
    gAlbedoSpec.rgb = albedoColor;
    
    // PBR properties
    float metallicValue = metallic;
    if (textureSize(metallicMap, 0).x > 1) {
        metallicValue *= texture(metallicMap, TexCoord).r;
    }
    
    float roughnessValue = roughness;
    if (textureSize(roughnessMap, 0).x > 1) {
        roughnessValue *= texture(roughnessMap, TexCoord).r;
    }
    
    float aoValue = ao;
    if (textureSize(aoMap, 0).x > 1) {
        aoValue *= texture(aoMap, TexCoord).r;
    }
    
    gMetallicRoughness = vec3(metallicValue, roughnessValue, aoValue);
    
    // Store specular component for Blinn-Phong fallback
    gAlbedoSpec.a = 1.0;
}
