#version 330 core

out vec4 FragColor;

in vec2 TexCoord;

// G-buffer textures
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gMetallicRoughnessAO;

uniform vec3 viewPos;

void main()
{
    // Retrieve data from G-buffer
    vec3 fragPos = texture(gPosition, TexCoord).rgb;
    vec3 normal = texture(gNormal, TexCoord).rgb;
    vec3 albedo = texture(gAlbedoSpec, TexCoord).rgb;
    
    // Check if this pixel has geometry
    if (length(normal) < 0.1) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Simple directional light
    vec3 lightDir = normalize(vec3(-1.0, -1.0, -1.0));
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    
    // Diffuse
    float diff = max(dot(normal, -lightDir), 0.0);
    vec3 diffuse = diff * lightColor * albedo;
    
    // Ambient
    vec3 ambient = 0.3 * albedo;
    
    // Final color
    vec3 result = ambient + diffuse;
    
    FragColor = vec4(result, 1.0);
}
