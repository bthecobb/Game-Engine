#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

// G-buffer textures
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gMetallicRoughness;

// Shadow mapping
uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

struct Light {
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float radius;
    float innerCutoff;
    float outerCutoff;
    int type; // 0: directional, 1: point, 2: spot
};

uniform Light light;
uniform vec3 viewPos;
uniform int debugMode; // 0=final, 1=position, 2=normal, 3=albedo, 4=metallic/roughness/AO
uniform float depthScale; // Runtime-adjustable depth scale for position visualization

float calculateShadow(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    if (projCoords.z > 1.0) return 0.0;
    
    float currentDepth = projCoords.z;
    float shadowBias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - shadowBias) > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    return shadow;
}

void main()
{
    vec3 fragPos = texture(gPosition, TexCoord).rgb;
    vec3 normal = texture(gNormal, TexCoord).rgb;
    vec3 albedo = texture(gAlbedoSpec, TexCoord).rgb;
    float specular = texture(gAlbedoSpec, TexCoord).a;
    vec3 metallicRoughness = texture(gMetallicRoughness, TexCoord).rgb;
    float metallic = metallicRoughness.r;
    float roughness = metallicRoughness.g;
    float ao = metallicRoughness.b;
    
    // Debug mode visualization
    if (debugMode == 1) {
        // Position buffer - runtime-adjustable Z-depth visualization
        float depth = abs(fragPos.z); // Use Z component only
        depth = depth / depthScale; // Use adjustable depth scale
        depth = clamp(depth, 0.0, 1.0);
        
        FragColor = vec4(vec3(depth), 1.0);
        return;
    }
    else if (debugMode == 2) {
        // Normal buffer (convert from -1,1 to 0,1 for visibility)
        FragColor = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    else if (debugMode == 3) {
        // Albedo buffer
        FragColor = vec4(albedo, 1.0);
        return;
    }
    else if (debugMode == 4) {
        // Metallic/Roughness/AO buffer
        FragColor = vec4(metallic, roughness, ao, 1.0);
        return;
    }

    // Simplified Lighting Without Shadows
    vec3 lightDir = normalize(-light.direction); // Assume directional for simplicity
    float NdotL = max(dot(normal, lightDir), 0.0);

    vec3 diffuseComponent = albedo * NdotL;
    vec3 specularComponent = vec3(0.0); // Simplified with no specular for now

    vec3 radiance = light.color * light.intensity;
    vec3 Lo = (diffuseComponent + specularComponent) * radiance;
    // Increased ambient from 0.03 to 0.15 for better scene visibility
    vec3 ambient = vec3(0.15 * albedo * ao);
    FragColor = vec4(ambient + Lo, 1.0);
}
