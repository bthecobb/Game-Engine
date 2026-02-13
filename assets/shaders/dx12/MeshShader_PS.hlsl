// MeshShader_PS.hlsl - Pixel Shader for Mesh Shader Pipeline
// Matches VertexOutput from MeshShader.hlsl and writes to G-Buffer

struct VertexOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 color : TEXCOORD3;
    nointerpolation uint meshletIndex : TEXCOORD4;
};

struct PSOutput
{
    float4 albedoRoughness : SV_TARGET0;  // RGB: Albedo, A: Roughness
    float4 normalMetallic : SV_TARGET1;   // RGB: Normal (view space), A: Metallic
    float4 emissiveAO : SV_TARGET2;       // RGB: Emissive, A: AO
    float2 velocity : SV_TARGET3;         // RG: Motion vectors (screen space)
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

// Helper: Encode normal to [0,1] range for storage
float3 EncodeNormal(float3 n)
{
    return n * 0.5 + 0.5;
}

PSOutput main(VertexOutput input)
{
    PSOutput output;
    
    // Albedo
    float3 albedo = albedoColor.rgb * input.color.rgb;
    
    // Normal (simple, no normal mapping for now)
    float3 worldNormal = normalize(input.normal);
    
    // Emissive
    float3 emissive = emissiveColor * emissiveStrength;
    emissive += albedo * input.color.rgb * input.color.a * 10.0;
    
    // Output
    output.albedoRoughness = float4(albedo, roughness);
    output.normalMetallic = float4(EncodeNormal(worldNormal), metallic);
    output.emissiveAO = float4(emissive, ambientOcclusion);
    output.velocity = float2(0.0, 0.0); // No motion vectors yet
    
    return output;
}
