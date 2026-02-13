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
    uint boneOffset;
    uint isSkinned;
    float2 _paddingObj; // Match C++ alignment if needed, or rely on pack. C++ had float _padding[14].
    // C++ struct: 3 mats (64*3=192) + 2 uints (8) + 56 bytes padding = 256.
    // HLSL cbuffer packing is 16-byte aligned vectors.
    // 3 mats take 3*64 = 192 bytes. Aligned.
    // boneOffset (4), isSkinned (4) = 8 bytes.
    // We need 56 bytes padding. float4 * 3 + float2 = 48 + 8 = 56.
    float4 _pad0; float4 _pad1; float4 _pad2; float2 _pad3; 
};

// Bone matrices buffer (t0 space1)
StructuredBuffer<float4x4> g_BoneMatrices : register(t0, space1);

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
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
    float4 color : COLOR;
};

VSOutput main(VSInput input) {
    VSOutput output;
    
    float4 localPos = float4(input.position, 1.0);
    float3 localNormal = input.normal;
    float3 localTangent = input.tangent;

    // Skinning logic (Always active in this shader)
    // Offset local bone index by instance's global offset
    uint4 globalIndices = uint4(input.boneIndices) + boneOffset;
    
    float4x4 boneTransform = g_BoneMatrices[globalIndices.x] * input.boneWeights.x;
    boneTransform += g_BoneMatrices[globalIndices.y] * input.boneWeights.y;
    boneTransform += g_BoneMatrices[globalIndices.z] * input.boneWeights.z;
    boneTransform += g_BoneMatrices[globalIndices.w] * input.boneWeights.w;
    
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
    
    output.bitangent = cross(output.normal, output.tangent);
    
    output.texcoord = input.texcoord;
    output.color = input.color;
    
    return output;
}
