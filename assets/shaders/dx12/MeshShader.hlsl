// MeshShader.hlsl - DX12 Ultimate Mesh Shader for AAA rendering
// Generates triangles from meshlets, outputs to pixel shader

// Meshlet structure (matches C++ Meshlet struct layout)
struct Meshlet {
    uint vertexOffset;
    uint vertexCount;
    uint primitiveOffset;
    uint primitiveCount;
    
    // Bounding info (interleaved in C++ struct)
    float4 sphere;
    float4 cone; 
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
StructuredBuffer<uint> PrimitiveIndices : register(t5); // Actually contains packed 8-bit indices!
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

// Helper: Unpack 8-bit index from 32-bit word buffer
uint UnpackPrimitiveIndex(uint byteOffset) {
    uint wordOffset = byteOffset / 4;
    uint shift = (byteOffset % 4) * 8;
    uint word = PrimitiveIndices[wordOffset];
    return (word >> shift) & 0xFF;
}

// Mesh shader entry point
[NumThreads(32, 1, 1)] // Adjusted to 32 to match AS wave size, loop handles >32
[OutputTopology("triangle")]
void main(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    in payload MeshPayload payload,
    out indices uint3 tris[MAX_PRIMITIVES],
    out vertices VertexOutput verts[MAX_VERTICES]
) {
    // Get meshlet index from payload
    uint meshletIndex = payload.meshletIndices[gid];
    
    // Get instance data (for world transform)
    InstanceData instance = Instances[payload.instanceId];
    
    // Load meshlet
    Meshlet m = Meshlets[meshletIndex];
    
    // Set output counts
    SetMeshOutputCounts(m.vertexCount, m.primitiveCount);
    
    // Load vertices (loop to handle >32 count if needed, or rely on 128 threads if I revert numthreads)
    // Actually, let's keep it robust for 64 vertices with 32 threads using a loop
    for (uint i = gtid; i < m.vertexCount; i += 32) {
        uint vertexIndex = VertexIndices[m.vertexOffset + i];
        
        // Load vertex attributes
        float3 localPos = Positions[vertexIndex];
        float3 normal = Normals[vertexIndex];
        float2 uv = UVs[vertexIndex];
        float4 color = Colors[vertexIndex];
        
        // Transform to world space
        float4 worldPos = mul(instance.worldMatrix, float4(localPos, 1.0f));
        
        // Output
        verts[i].position = mul(viewProjection, worldPos);
        verts[i].worldPos = worldPos.xyz;
        verts[i].normal = mul((float3x3)instance.normalMatrix, normal);
        verts[i].uv = uv;
        verts[i].color = color;
        verts[i].meshletIndex = meshletIndex;
    }
    
    // Load primitives
    for (uint i = gtid; i < m.primitiveCount; i += 32) {
        uint primOffset = m.primitiveOffset + i * 3;
        
        // Unpack 3 indices (8-bit each)
        uint i0 = UnpackPrimitiveIndex(primOffset + 0);
        uint i1 = UnpackPrimitiveIndex(primOffset + 1);
        uint i2 = UnpackPrimitiveIndex(primOffset + 2);
        
        tris[i] = uint3(i0, i1, i2);
    }
}
