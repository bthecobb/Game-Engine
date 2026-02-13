// Forward Rendering Pixel Shader - Single Target Output
// Simplified shader that outputs color directly to the swap chain

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BITANGENT;
    float2 texcoord : TEXCOORD;
    float4 currClipPos : CURR_CLIP_POS;
    float4 prevClipPos : PREV_CLIP_POS;
};

// Material properties
cbuffer MaterialConstants : register(b2)
{
    float4 albedoColor;      // RGB: color, A: unused
    float roughness;
    float metallic;
    float ambientOcclusion;
    float emissiveStrength;
    float3 emissiveColor;
    float _padding;
};

cbuffer PerFrameConstants : register(b0)
{
    column_major float4x4 viewMatrix;
    column_major float4x4 projMatrix;
    column_major float4x4 viewProjMatrix;
    column_major float4x4 prevViewProjMatrix;
    float3 cameraPosition;
    float time;
    float deltaTime;
    float3 _padding0;
};

float4 main(PSInput input) : SV_TARGET
{
    // Simple forward rendering using material albedo and a single directional light.
    float3 albedo = albedoColor.rgb;
    float3 normal = normalize(input.normal);

    // Simple directional light from above/right.
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float NdotL = max(0.0, dot(normal, lightDir));

    // Ambient + diffuse.
    float3 ambient = albedo * 0.3;
    float3 diffuse = albedo * NdotL * 0.7;
    float3 finalColor = ambient + diffuse;

    // Add emissive.
    finalColor += emissiveColor * emissiveStrength;

    return float4(finalColor, 1.0);
}
