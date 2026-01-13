// MeshShader.hlsl - DX12 Ultimate Mesh Shader for AAA rendering
// Generates triangles from meshlets, outputs to pixel shader

// Meshlet structure (matches C++ Meshlet struct)
struct Meshlet {
    uint vertexOffset;
    uint vertexCount;
    uint primitiveOffset;
    uint primitiveCount;
};

// Meshlet bounds for culling
struct MeshletBounds {
    float4 sphere;  // xyz = center, w = radius
    float4 cone;    // xyz = axis, w = cos(angle)
};

// Per-object instance data
struct InstanceData {
    float4x4 worldMatrix;
    float4x4 normalMatrix;
    uint meshletOffset;
    uint meshletCount;
    uint2 padding;
};

// Vertex output to pixel shader
struct VertexOutput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 color : TEXCOORD3;
    nointerpolation uint meshletIndex : TEXCOORD4;
};

// Constant buffers
cbuffer CameraBuffer : register(b0) {
    float4x4 viewProjection;
    float4x4 view;
    float3 cameraPosition;
    float padding;
};

// Structured buffers (bindless pattern)
StructuredBuffer<float3> Positions : register(t0);
StructuredBuffer<float3> Normals : register(t1);
StructuredBuffer<float2> UVs : register(t2);
StructuredBuffer<float4> Colors : register(t3);
StructuredBuffer<uint> VertexIndices : register(t4);
StructuredBuffer<uint> PrimitiveIndices : register(t5);
StructuredBuffer<Meshlet> Meshlets : register(t6);
StructuredBuffer<InstanceData> Instances : register(t7);

// Mesh shader constants
#define MAX_VERTICES 64
#define MAX_PRIMITIVES 126

// Payload from amplification shader
struct MeshPayload {
    uint meshletIndices[32];  // Which meshlets to render (from AS)
    uint instanceId;
};

// Mesh shader entry point
[NumThreads(128, 1, 1)]
[OutputTopology("triangle")]
void main(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    in payload MeshPayload meshPayload,
    out vertices VertexOutput verts[MAX_VERTICES],
    out indices uint3 tris[MAX_PRIMITIVES]
) {
    // Get meshlet index from payload
    uint meshletIndex = meshPayload.meshletIndices[gid % 32];
    Meshlet meshlet = Meshlets[meshletIndex];
    InstanceData instance = Instances[meshPayload.instanceId];
    
    // Set mesh output counts
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.primitiveCount);
    
    // Each thread processes one vertex
    if (gtid < meshlet.vertexCount) {
        uint vertexIndex = VertexIndices[meshlet.vertexOffset + gtid];
        
        float3 localPos = Positions[vertexIndex];
        float3 localNormal = Normals[vertexIndex];
        
        // Transform to world space
        float4 worldPos = mul(instance.worldMatrix, float4(localPos, 1.0));
        float3 worldNormal = normalize(mul((float3x3)instance.normalMatrix, localNormal));
        
        // Output vertex
        VertexOutput v;
        v.position = mul(viewProjection, worldPos);
        v.worldPos = worldPos.xyz;
        v.normal = worldNormal;
        v.uv = UVs[vertexIndex];
        v.color = Colors[vertexIndex];
        v.meshletIndex = meshletIndex;
        
        verts[gtid] = v;
    }
    
    // Each thread also processes primitives (indices)
    if (gtid < meshlet.primitiveCount) {
        // Primitive indices are packed as 3 bytes per triangle
        uint primOffset = meshlet.primitiveOffset + gtid * 3;
        uint i0 = PrimitiveIndices[primOffset + 0];
        uint i1 = PrimitiveIndices[primOffset + 1];
        uint i2 = PrimitiveIndices[primOffset + 2];
        
        tris[gtid] = uint3(i0, i1, i2);
    }
}
