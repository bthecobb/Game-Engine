// Geometry Pass Pixel Shader - AAA G-Buffer Output
// Writes to 4 render targets + depth:
// RT0: Albedo (RGB) + Roughness (A)
// RT1: Normal (RGB) + Metallic (A)
// RT2: Emissive (RGB) + AO (A)
// RT3: Velocity (RG)

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

// Textures (optional)
Texture2D albedoTex : register(t0);
Texture2D normalTex : register(t1);
Texture2D roughnessTex : register(t2);
Texture2D metallicTex : register(t3);
Texture2D aoTex : register(t4);
Texture2D emissiveTex : register(t5);

SamplerState linearSampler : register(s0);

// Helper: Encode normal to [0,1] range for storage
float3 EncodeNormal(float3 n)
{
    return n * 0.5 + 0.5;
}

PSOutput main(PSInput input)
{
    PSOutput output;
    
    // Sample albedo texture or use constant
    float3 albedo = albedoColor.rgb;
    // Uncomment when textures are bound:
    // albedo *= albedoTex.Sample(linearSampler, input.texcoord).rgb;
    
    // Sample roughness
    float roughnessValue = roughness;
    // roughnessValue *= roughnessTex.Sample(linearSampler, input.texcoord).r;
    
    // Sample metallic
    float metallicValue = metallic;
    // metallicValue *= metallicTex.Sample(linearSampler, input.texcoord).r;
    
    // Sample AO
    float aoValue = ambientOcclusion;
    // aoValue *= aoTex.Sample(linearSampler, input.texcoord).r;
    
    // Sample and transform normal map (if available)
    float3 worldNormal = normalize(input.normal);
    // float3 normalMap = normalTex.Sample(linearSampler, input.texcoord).xyz * 2.0 - 1.0;
    // float3x3 TBN = float3x3(normalize(input.tangent), normalize(input.bitangent), worldNormal);
    // worldNormal = normalize(mul(normalMap, TBN));
    
    // Sample emissive
    float3 emissive = emissiveColor * emissiveStrength;
    // emissive *= emissiveTex.Sample(linearSampler, input.texcoord).rgb;
    
    // Calculate motion vectors for DLSS/TAA
    // Project current and previous positions to screen space
    float2 currScreen = input.currClipPos.xy / input.currClipPos.w;
    float2 prevScreen = input.prevClipPos.xy / input.prevClipPos.w;
    float2 motionVector = (currScreen - prevScreen) * float2(0.5, -0.5);  // Screen space velocity
    
    // Output to G-Buffer
    output.albedoRoughness = float4(albedo, roughnessValue);
    output.normalMetallic = float4(EncodeNormal(worldNormal), metallicValue);
    output.emissiveAO = float4(emissive, aoValue);
    output.velocity = motionVector;
    
    return output;
}
