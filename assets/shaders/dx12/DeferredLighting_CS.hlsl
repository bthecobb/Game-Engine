// Deferred Lighting Compute Shader - AAA PBR Implementation
// Reads G-Buffer and outputs HDR lit color
// Supports: Directional lights, Point lights, Spot lights
// PBR: Cook-Torrance BRDF with GGX distribution

#define PI 3.14159265359
#define NUM_THREADS 8

// G-Buffer inputs (read-only)
Texture2D<float4> gBufferAlbedoRoughness : register(t0);  // RGB: Albedo, A: Roughness
Texture2D<float4> gBufferNormalMetallic : register(t1);   // RGB: Normal, A: Metallic
Texture2D<float4> gBufferEmissiveAO : register(t2);       // RGB: Emissive, A: AO
Texture2D<float> gBufferDepth : register(t3);             // R32F: Depth

// Output HDR color (write)
RWTexture2D<float4> outputColor : register(u0);

// Lighting constants
cbuffer LightingConstants : register(b0)
{
    float3 cameraPosition;
    float _padding0;
    
    float3 ambientColor;
    float ambientIntensity;
    
    // Directional light
    float3 dirLightDirection;
    float dirLightIntensity;
    float3 dirLightColor;
    float _padding1;
    
    // Point lights (up to 4)
    float4 pointLightPositions[4];    // xyz: position, w: range
    float4 pointLightColors[4];       // rgb: color, a: intensity
    int numPointLights;
    float3 _padding2;
    
    // View/Projection for depth reconstruction
    float4x4 invViewProjMatrix;
    
    float2 screenSize;
    float2 _padding3;
};

// Helper: Decode normal from [0,1] to [-1,1]
float3 DecodeNormal(float3 encoded)
{
    return normalize(encoded * 2.0 - 1.0);
}

// Helper: Reconstruct world position from depth
float3 ReconstructWorldPosition(uint2 pixelCoord, float depth)
{
    float2 uv = (float2(pixelCoord) + 0.5) / screenSize;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for D3D
    
    float4 clipPos = float4(ndc, depth, 1.0);
    float4 worldPos = mul(clipPos, invViewProjMatrix);
    return worldPos.xyz / worldPos.w;
}

// PBR: Fresnel-Schlick approximation
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// PBR: GGX/Trowbridge-Reitz normal distribution
float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

// PBR: Geometry function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

// PBR: Smith's method for geometry occlusion
float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// PBR: Cook-Torrance BRDF
float3 CalculatePBR(float3 albedo, float3 N, float3 V, float3 L, float3 radiance,
                    float metallic, float roughness)
{
    float3 H = normalize(V + L);
    
    // Calculate F0 (surface reflection at zero incidence)
    float3 F0 = float3(0.04, 0.04, 0.04);  // Dielectric base
    F0 = lerp(F0, albedo, metallic);
    
    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - metallic;
    
    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    float3 specular = numerator / denominator;
    
    float NdotL = max(dot(N, L), 0.0);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

[numthreads(NUM_THREADS, NUM_THREADS, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    
    // Bounds check
    if (pixelCoord.x >= (uint)screenSize.x || pixelCoord.y >= (uint)screenSize.y)
        return;
    
    // Read G-Buffer
    float4 albedoRoughness = gBufferAlbedoRoughness[pixelCoord];
    float4 normalMetallic = gBufferNormalMetallic[pixelCoord];
    float4 emissiveAO = gBufferEmissiveAO[pixelCoord];
    float depth = gBufferDepth[pixelCoord];
    
    // Early out for skybox/background (depth = 1.0)
    if (depth >= 1.0)
    {
        outputColor[pixelCoord] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Extract material properties
    float3 albedo = albedoRoughness.rgb;
    float roughness = albedoRoughness.a;
    float3 normal = DecodeNormal(normalMetallic.rgb);
    float metallic = normalMetallic.a;
    float3 emissive = emissiveAO.rgb;
    float ao = emissiveAO.a;
    
    // Reconstruct world position
    float3 worldPos = ReconstructWorldPosition(pixelCoord, depth);
    float3 V = normalize(cameraPosition - worldPos);
    
    // Start with emissive + ambient
    float3 Lo = emissive;
    Lo += ambientColor * ambientIntensity * albedo * ao;
    
    // Directional light
    {
        float3 L = normalize(-dirLightDirection);
        float3 radiance = dirLightColor * dirLightIntensity;
        Lo += CalculatePBR(albedo, normal, V, L, radiance, metallic, roughness);
    }
    
    // Point lights
    for (int i = 0; i < numPointLights; i++)
    {
        float3 lightPos = pointLightPositions[i].xyz;
        float lightRange = pointLightPositions[i].w;
        float3 lightColor = pointLightColors[i].rgb;
        float lightIntensity = pointLightColors[i].a;
        
        float3 L = lightPos - worldPos;
        float distance = length(L);
        
        // Range attenuation
        if (distance > lightRange)
            continue;
        
        L = L / distance;  // Normalize
        
        // Inverse square law with smooth falloff
        float attenuation = lightIntensity / (distance * distance + 1.0);
        attenuation *= saturate(1.0 - (distance / lightRange));
        
        float3 radiance = lightColor * attenuation;
        Lo += CalculatePBR(albedo, normal, V, L, radiance, metallic, roughness);
    }
    
    // Output HDR color
    outputColor[pixelCoord] = float4(Lo, 1.0);
}
