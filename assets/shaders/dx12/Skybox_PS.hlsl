// Skybox_PS.hlsl - Procedural atmospheric sky pixel shader
// Features: Sky gradient, sun disk, horizon haze

cbuffer SkyConstants : register(b1)
{
    float3 sunDirection;    // Normalized sun direction (towards sun)
    float sunIntensity;     // Sun brightness multiplier
    
    float3 zenithColor;     // Sky color at top (deep blue)
    float horizonBlend;     // How much to blend to horizon
    
    float3 horizonColor;    // Sky color at horizon (light blue/orange)
    float sunSize;          // Angular size of sun disk
    
    float3 sunColor;        // Color of the sun
    float exposure;         // HDR exposure
}

struct PSInput
{
    float4 Position : SV_POSITION;
    float3 ViewDir : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 viewDir = normalize(input.ViewDir);
    
    // Default sky parameters (can be overridden by cbuffer)
    float3 skyZenith = float3(0.2, 0.4, 0.8);      // Deep blue at top
    float3 skyHorizon = float3(0.6, 0.8, 1.0);     // Light blue at horizon
    float3 groundColor = float3(0.3, 0.25, 0.2);   // Brown ground
    float3 sunDir = normalize(float3(0.5, 0.3, 0.8)); // Sun position
    float3 sunCol = float3(1.0, 0.95, 0.8);        // Warm white sun
    
    // Calculate sky gradient based on vertical angle
    float upDot = viewDir.y; // -1 = down, 0 = horizon, 1 = up
    
    // Sky gradient: zenith to horizon
    float skyGradient = saturate(upDot);
    float3 skyColor = lerp(skyHorizon, skyZenith, pow(skyGradient, 0.5));
    
    // Add horizon haze (brighter near horizon)
    float horizonHaze = 1.0 - abs(upDot);
    horizonHaze = pow(horizonHaze, 4.0);
    skyColor = lerp(skyColor, skyHorizon * 1.2, horizonHaze * 0.3);
    
    // Sun disk
    float sunDot = dot(viewDir, sunDir);
    float sunDisk = smoothstep(0.995, 0.999, sunDot); // Sharp sun disk
    float sunGlow = pow(saturate(sunDot), 32.0) * 0.5; // Soft glow around sun
    
    // Add sun to sky
    skyColor += sunCol * (sunDisk * 10.0 + sunGlow);
    
    // Below horizon - fade to ground color
    if (upDot < 0.0)
    {
        float groundBlend = saturate(-upDot * 2.0);
        skyColor = lerp(skyHorizon, groundColor, groundBlend);
    }
    
    // Simple tone mapping
    skyColor = skyColor / (skyColor + 1.0);
    
    // Gamma correction
    skyColor = pow(skyColor, 1.0 / 2.2);
    
    return float4(skyColor, 1.0);
}
