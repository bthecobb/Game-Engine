// Post-process fullscreen triangle VS (no vertex buffers)
// Outputs clip-space position only; pixel shader uses SV_POSITION to compute pixel coords.

struct VSOut
{
    float4 position : SV_POSITION;
};

VSOut main(uint vertexId : SV_VertexID)
{
    VSOut o;

    // Fullscreen triangle in clip space
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0)
    };

    o.position = float4(positions[vertexId], 0.0, 1.0);
    return o;
}
