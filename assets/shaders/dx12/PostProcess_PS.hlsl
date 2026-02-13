// Post-process pixel shader
// Reads HDR input color buffer and writes tone-mapped SDR to swapchain.
// Assumes input texture resolution matches the current viewport/backbuffer.

Texture2D<float4> InputColor : register(t0);

// Simple Reinhard tone mapping + gamma
float3 ToneMap(float3 hdrColor)
{
    float3 mapped = hdrColor / (1.0 + hdrColor);
    mapped = pow(mapped, 1.0 / 2.2);
    return mapped;
}

float4 main(float4 position : SV_POSITION) : SV_TARGET
{
    // SV_POSITION.xy is in pixel coordinates (after viewport transform)
    uint2 pixel = uint2(position.xy);
    float4 hdr = InputColor.Load(int3(pixel, 0));

    float3 ldr = ToneMap(hdr.rgb);
    return float4(ldr, 1.0);
}
