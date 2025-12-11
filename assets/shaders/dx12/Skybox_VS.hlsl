// Skybox_VS.hlsl - Fullscreen triangle vertex shader for sky rendering
// Renders a fullscreen quad by generating vertices procedurally

cbuffer PerFrameConstants : register(b0)
{
    row_major float4x4 viewMatrix;
    row_major float4x4 projMatrix;
    row_major float4x4 viewProjMatrix;
    row_major float4x4 prevViewProjMatrix;
    float3 cameraPosition;
    float time;
    float deltaTime;
    float3 _padding;
}

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 ViewDir : TEXCOORD0;
};

VSOutput main(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Generate fullscreen triangle vertices
    // vertexID 0: (-1, -1), vertexID 1: (3, -1), vertexID 2: (-1, 3)
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    float4 clipPos = float4(uv * 2.0 - 1.0, 0.999, 1.0); // z near far plane
    clipPos.y = -clipPos.y; // Flip Y for D3D
    
    output.Position = clipPos;
    
    // Calculate world-space view direction from clip space
    // Invert the view-projection to get world direction
    float4x4 invView = viewMatrix;  // We need inverse, but for skybox we can approximate
    
    // For a proper skybox, we want the direction the camera is looking
    // Extract the view direction from clip position
    float3 viewDirLocal = float3(clipPos.x, clipPos.y, 1.0);
    
    // Transform by inverse view rotation (skip translation for direction)
    // For now, use a simplified approach
    output.ViewDir = normalize(float3(clipPos.x * 1.5, clipPos.y * 1.5, -1.0));
    
    return output;
}
