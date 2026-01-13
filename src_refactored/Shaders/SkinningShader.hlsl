// #include "Common.hlsli"

cbuffer PerFrameConstants : register(b0) {
    column_major float4x4 viewMatrix;
    column_major float4x4 projMatrix;
    column_major float4x4 viewProjMatrix;
    column_major float4x4 prevViewProjMatrix;
    float3 cameraPosition;
    float time;
    float deltaTime;
    float3 _padding;
};

cbuffer PerObjectConstants : register(b1) {
    column_major float4x4 worldMatrix;
    column_major float4x4 prevWorldMatrix;
    column_major float4x4 normalMatrix;
    // Extra data not in GeometryPass_VS but used here? 
    // Wait, if I change this, I might break alignment if the C++ side sends more data.
    // The C++ side sends PerObjectConstants which has: world, prevWorld, normal.
    // The extra fields (color, roughness, etc.) are in MaterialConstants (b2) in the C++ code!
    // SkinningShader was trying to read them from b1.
    // I should remove them from here and rely on the PS to read them from b2 if needed.
    // BUT, SkinningShader uses 'useSkinning' which was in b1.
    // I need to check where 'useSkinning' is sent from C++.
};

// Bone matrices buffer (t0 space1)
StructuredBuffer<float4x4> g_BoneMatrices : register(t0, space1);

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 texcoord : TEXCOORD;
    int4 boneIndices : BLENDINDICES;
    float4 boneWeights : BLENDWEIGHT;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BITANGENT;
    float2 texcoord : TEXCOORD;
    float4 currClipPos : CURR_CLIP_POS;
    float4 prevClipPos : PREV_CLIP_POS;
};

VSOutput main(VSInput input) {
    VSOutput output;
    
    float4 localPos = float4(input.position, 1.0);
    float3 localNormal = input.normal;
    float3 localTangent = input.tangent;

    // Skinning logic
    // We assume this shader is ONLY used for skinned meshes, so we always skin.
    // If useSkinning flag is needed, it must be passed correctly.
    // However, checking C++ code, we switch PSO based on skinning.
    // So this shader IS the skinning shader. We don't need a flag.
    
    float4x4 boneTransform = g_BoneMatrices[input.boneIndices.x] * input.boneWeights.x;
    boneTransform += g_BoneMatrices[input.boneIndices.y] * input.boneWeights.y;
    boneTransform += g_BoneMatrices[input.boneIndices.z] * input.boneWeights.z;
    boneTransform += g_BoneMatrices[input.boneIndices.w] * input.boneWeights.w;
    
    localPos = mul(boneTransform, localPos);
    localNormal = mul((float3x3)boneTransform, localNormal);
    localTangent = mul((float3x3)boneTransform, localTangent);

    // Transform to world space
    float4 worldPos = mul(worldMatrix, localPos);
    output.worldPos = worldPos.xyz;
    
    // Transform to clip space
    output.position = mul(viewProjMatrix, worldPos);
    
    // Motion vectors
    output.currClipPos = output.position;
    // For now, use current position for previous to avoid artifacts (no motion blur for skinned yet)
    output.prevClipPos = output.position; 
    
    // Normal/Tangent
    output.normal = normalize(mul((float3x3)normalMatrix, localNormal));
    output.tangent = normalize(mul((float3x3)normalMatrix, localTangent));
    
    // Gram-Schmidt
    output.tangent = normalize(output.tangent - dot(output.tangent, output.normal) * output.normal);
    
    // Bitangent
    output.bitangent = cross(output.normal, output.tangent);
    
    output.texcoord = input.texcoord;
    
    return output;
}
