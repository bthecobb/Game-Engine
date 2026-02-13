// ParticleMeshShader.hlsl
// Expands points from StructuredBuffer into view-facing quads

// #include "Common.hlsli" (Inlined below)

struct PerFrameConstants {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4x4 prevViewProj;
    float3 cameraPos;
    float time;
    float deltaTime;
    float3 _padding;
};

struct Particle {
    float3 position;
    float life;
    float3 velocity;
    float size;
    float4 color;
};

StructuredBuffer<Particle> g_Particles : register(t0);
ConstantBuffer<PerFrameConstants> g_Frame : register(b0);

struct VertexOut {
    float4 posH : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void main(
    uint gtid : SV_GroupThreadID,
    uint dtid : SV_DispatchThreadID,
    out indices uint3 tris[64],
    out vertices VertexOut verts[128]
)
{
    // Culling
    // For now draw everything valid
    
    // Each thread processes one particle -> 4 vertices, 2 triangles
    // Need allocation logic if culling.
    // Mesh Shader usually processes chunk (Meshlet).
    // Here we treat Group as a Meshlet of 32 particles.
    
    uint particleIdx = dtid; // Global index? 
    // Mesh Shader semantics: Dispatch(Count/32).
    // Wait, dtid depends on Dispatch.
    
    SetMeshOutputCounts(128, 64); // 32 particles * 4 verts, 32 * 2 tris
    
    if (particleIdx >= 100000) return; // Bounds check (hardcoded limit for now or pass in CB)
    
    Particle p = g_Particles[particleIdx];
    
    if (p.life <= 0.0) {
        // Degenerate
        // Actually we should compact before this or output 0 primitives.
        // For simplicity, output degenerate triangles (0 area)
        for (int i=0; i<4; ++i) verts[gtid*4+i].posH = float4(0,0,0,0);
        return;
    }
    
    // Billboard Expansion
    float3 center = p.position;
    float s = p.size * 0.5;
    float4 color = p.color;
    
    // Camera Basis (View Space aligned)
    // We want camera up/right
    float3 camRight = float3(g_Frame.view[0][0], g_Frame.view[1][0], g_Frame.view[2][0]);
    float3 camUp = float3(g_Frame.view[0][1], g_Frame.view[1][1], g_Frame.view[2][1]);
    
    float3 p0 = center - camRight * s + camUp * s; // TL
    float3 p1 = center + camRight * s + camUp * s; // TR
    float3 p2 = center - camRight * s - camUp * s; // BL
    float3 p3 = center + camRight * s - camUp * s; // BR
    
    // Project
    verts[gtid*4+0].posH = mul(float4(p0, 1.0), g_Frame.viewProj);
    verts[gtid*4+0].uv = float2(0, 0);
    verts[gtid*4+0].color = color;
    
    verts[gtid*4+1].posH = mul(float4(p1, 1.0), g_Frame.viewProj);
    verts[gtid*4+1].uv = float2(1, 0);
    verts[gtid*4+1].color = color;
    
    verts[gtid*4+2].posH = mul(float4(p2, 1.0), g_Frame.viewProj);
    verts[gtid*4+2].uv = float2(0, 1);
    verts[gtid*4+2].color = color;
    
    verts[gtid*4+3].posH = mul(float4(p3, 1.0), g_Frame.viewProj);
    verts[gtid*4+3].uv = float2(1, 1);
    verts[gtid*4+3].color = color;
    
    // Indices (Clockwise)
    // 0-1
    // |/|
    // 2-3
    // Tri 1: 0, 1, 2
    // Tri 2: 2, 1, 3
    tris[gtid*2+0] = uint3(gtid*4+0, gtid*4+1, gtid*4+2);
    tris[gtid*2+1] = uint3(gtid*4+2, gtid*4+1, gtid*4+3);
}
