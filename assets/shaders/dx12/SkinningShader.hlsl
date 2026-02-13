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

// Output must match PSInput in GeometryPass_PS.hlsl (and GeometryPass_VS.hlsl output)
struct VSOutput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BITANGENT;
    float2 texcoord : TEXCOORD;
    float4 currClipPos : CURR_CLIP_POS;
    float4 prevClipPos : PREV_CLIP_POS;
    float4 vertexColor : COLOR0;
    // float3 viewDir : TEXCOORD1; // Not used in G-Buffer pass usually, removing to match GeometryPass_VS
};

VSOutput main(VSInput input) {
    VSOutput output;
    
    // Default to static
    float3 localPos = input.position;
    float3 localNormal = input.normal;
    float3 localTangent = input.tangent;

    // Apply Skinning
    if (isSkinned) {
        float4x4 boneTransform = {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };
        
        // Sum weighted bone matrices
        // Unroll explicitly if needed, but loop is fine usually. Here explicit for clarity.
        boneTransform += g_BoneMatrices[input.boneIndices.x] * input.boneWeights.x;
        boneTransform += g_BoneMatrices[input.boneIndices.y] * input.boneWeights.y;
        boneTransform += g_BoneMatrices[input.boneIndices.z] * input.boneWeights.z;
        boneTransform += g_BoneMatrices[input.boneIndices.w] * input.boneWeights.w;
        
        // Apply transform
        localPos = mul(boneTransform, float4(localPos, 1.0f)).xyz;
        localNormal = mul((float3x3)boneTransform, localNormal);
        localTangent = mul((float3x3)boneTransform, localTangent);
    }

    // Transform position to world space
    float4 worldPos4 = mul(worldMatrix, float4(localPos, 1.0f));
    output.worldPos = worldPos4.xyz;
    
    // Transform to clip space
    output.position = mul(viewProjMatrix, worldPos4);
    
    // Motion vectors: Current and previous clip space positions
    output.currClipPos = output.position;
    
    // For previous position, we need previous frame bone transforms if we want accurate skinned motion blur.
    // For now, checking if we handle static motion or ignore skinning for prev pos.
    // Ideally we'd have g_PrevBoneMatrices. 
    // Fallback: Use current skinning with previous world matrix, or just static.
    // Using simple static logic for prev pos to avoid artifacts for now:
    float4 prevWorldPos = mul(prevWorldMatrix, float4(localPos, 1.0f)); // Using current localPos is technically wrong if anim changed, but better than nothing.
    output.prevClipPos = mul(prevViewProjMatrix, prevWorldPos);
    
    // Normal/Tangent to world space
    // Note: casts to float3x3 might need transpose if matrices are column-major but multiplication order expects row vectors? 
    // HLSL mul(vector, matrix) treats vector as row.
    // normalMatrix is usually InverseTranspose(World).
    output.normal = normalize(mul((float3x3)normalMatrix, localNormal));
    output.tangent = normalize(mul((float3x3)normalMatrix, localTangent));
    
    // Gram-Schmidt
    output.tangent = normalize(output.tangent - dot(output.tangent, output.normal) * output.normal);
    
    // Bitangent
    output.bitangent = cross(output.normal, output.tangent);
    
    output.texcoord = input.texcoord;
    output.vertexColor = input.color;
    
    return output;
}
