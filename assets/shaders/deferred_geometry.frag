#version 330 core

layout (location = 0) out vec3 gPosition;      // World position
layout (location = 1) out vec3 gNormal;        // World normal
layout (location = 2) out vec4 gAlbedoSpec;    // Albedo + specular
layout (location = 3) out vec4 gMetallicRoughnessAOEmissive; // Metallic + Roughness + AO + Emissive Power
layout (location = 4) out vec3 gEmissive;      // Emissive color (glowing elements)

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 VertexColor;
in vec3 Emissive;
in vec3 Tangent;
in vec3 Bitangent;
in vec4 FragPosLightSpace;

// PBR Material uniforms
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D emissiveMap;  // Emissive texture for glowing windows

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;
uniform float emissiveIntensity;  // Global emissive multiplier

// Debug: force emissive output for validation (1 = on, 0 = off)
uniform int debugForceEmissive;

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
    
    // Albedo - blend with vertex color
    vec3 albedoColor = albedo;
    if (textureSize(albedoMap, 0).x > 1) {
        albedoColor *= texture(albedoMap, TexCoord).rgb;
    }
    // If vertex color is non-zero, use it (for procedural buildings)
    if (length(VertexColor) > 0.01) {
        albedoColor = VertexColor;
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
    
    // Sample emissive texture for glowing windows
    vec3 emissiveColor = vec3(0.0);
    float emissivePower = 0.0;
    
    if (textureSize(emissiveMap, 0).x > 1) {
        // Sample emissive texture: RGB=color, A=intensity
        vec4 emissiveSample = texture(emissiveMap, TexCoord);
        emissiveColor = emissiveSample.rgb;
        emissivePower = emissiveSample.a * emissiveIntensity * 15.0;  // Scale to visible intensity
    } else {
        // Fallback to vertex emissive (for backwards compatibility)
        emissivePower = max(max(Emissive.r, Emissive.g), Emissive.b);
        emissiveColor = (emissivePower > 0.01) ? (Emissive / emissivePower) : vec3(0.0);
    }
    
    gMetallicRoughnessAOEmissive = vec4(metallicValue, roughnessValue, aoValue, emissivePower);
    gEmissive = emissiveColor;

    // Optional debug: force bright emissive to validate attachment/index wiring
    if (debugForceEmissive == 1) {
        gEmissive = vec3(10.0);
        gMetallicRoughnessAOEmissive.a = 15.0;
    }
    
    // Store specular component for Blinn-Phong fallback
    gAlbedoSpec.a = 1.0;
}
