// Geometry Pass Vertex Shader - AAA G-Buffer Rendering
// Outputs to multiple render targets: Albedo, Normal, Velocity

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
    int4   boneIndices : BLENDINDICES;
    float4 boneWeights : BLENDWEIGHT;
};  // rgb = vertex color, a = emissive intensity

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BITANGENT;
    float2 texcoord : TEXCOORD;
    float4 currClipPos : CURR_CLIP_POS;  // Current frame clip space
    float4 prevClipPos : PREV_CLIP_POS;  // Previous frame clip space (for motion vectors)
    float4 vertexColor : COLOR0;         // Passed from vertex shader
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
    float3 _padding;
};

cbuffer PerObjectConstants : register(b1)
{
    column_major float4x4 worldMatrix;
    column_major float4x4 prevWorldMatrix;  // For motion vectors
    column_major float4x4 normalMatrix;     // Transpose(Inverse(worldMatrix))
};

PSInput main(VSInput input)
{
    PSInput output;
    
    // Transform position to world space (column-major: matrix * vector)
    float4 worldPos = mul(worldMatrix, float4(input.position, 1.0));
    output.worldPos = worldPos.xyz;
    
    // Transform to clip space
    output.position = mul(viewProjMatrix, worldPos);
    
    // Current and previous clip space positions for motion vectors
    output.currClipPos = output.position;
    float4 prevWorldPos = mul(prevWorldMatrix, float4(input.position, 1.0));
    output.prevClipPos = mul(prevViewProjMatrix, prevWorldPos);
    
    // Transform normal, tangent to world space
    output.normal = normalize(mul(normalMatrix, float4(input.normal, 0.0)).xyz);
    output.tangent = normalize(mul(normalMatrix, float4(input.tangent, 0.0)).xyz);
    
    // Gram-Schmidt orthogonalize tangent with respect to normal
    output.tangent = normalize(output.tangent - dot(output.tangent, output.normal) * output.normal);
    
    // Calculate bitangent
    output.bitangent = cross(output.normal, output.tangent);
    
    // Pass through texcoords
    output.texcoord = input.texcoord;
    
    // Pass through vertex color (for procedural buildings with window lights)
    output.vertexColor = input.color;
    
    return output;
}
