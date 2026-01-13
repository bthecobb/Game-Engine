// AmplificationShader.hlsl - DX12 Ultimate Amplification Shader
// Per-meshlet culling: frustum + backface, dispatches visible meshlets to mesh shader

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

// Payload to mesh shader
struct MeshPayload {
    uint meshletIndices[32];
    uint instanceId;
};

// Camera data
cbuffer CameraBuffer : register(b0) {
    float4x4 viewProjection;
    float4x4 view;
    float3 cameraPosition;
    float padding;
    float4 frustumPlanes[6];  // Frustum planes for culling
};

// Buffers
StructuredBuffer<MeshletBounds> MeshletBoundsBuffer : register(t0);
StructuredBuffer<InstanceData> Instances : register(t1);

// Root constants
cbuffer RootConstants : register(b1) {
    uint instanceId;
    uint meshletOffset;
    uint meshletCount;
    uint padding2;
};

// Wave-level visible count
groupshared uint gs_visibleCount;
groupshared uint gs_visibleMeshlets[32];

// Test sphere against frustum planes
bool FrustumCullSphere(float4 sphere) {
    float3 center = sphere.xyz;
    float radius = sphere.w;
    
    [unroll]
    for (uint i = 0; i < 6; ++i) {
        float4 plane = frustumPlanes[i];
        float dist = dot(plane.xyz, center) + plane.w;
        if (dist < -radius) {
            return true;  // Culled
        }
    }
    return false;  // Visible
}

// Test normal cone for backface culling
bool ConeCull(float4 cone, float3 cameraDir) {
    float3 axis = cone.xyz;
    float cosAngle = cone.w;
    
    // If cone angle >= 90 degrees, can't cull
    if (cosAngle <= 0.0) return false;
    
    // Check if camera is outside the cone
    float viewDot = dot(axis, -cameraDir);
    return viewDot > cosAngle;
}

// Amplification shader - one thread per meshlet
[NumThreads(32, 1, 1)]
void main(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID
) {
    // Initialize shared memory
    if (gtid == 0) {
        gs_visibleCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Get instance and meshlet info
    InstanceData instance = Instances[instanceId];
    uint globalMeshletIndex = instance.meshletOffset + gid * 32 + gtid;
    
    bool visible = false;
    if (gtid < meshletCount && globalMeshletIndex < (instance.meshletOffset + instance.meshletCount)) {
        MeshletBounds bounds = MeshletBoundsBuffer[globalMeshletIndex];
        
        // Transform sphere to world space
        float4 worldSphere;
        worldSphere.xyz = mul(instance.worldMatrix, float4(bounds.sphere.xyz, 1.0)).xyz;
        worldSphere.w = bounds.sphere.w;  // Assume uniform scale
        
        // Frustum cull
        if (!FrustumCullSphere(worldSphere)) {
            // Backface cone cull
            float3 cameraDir = normalize(worldSphere.xyz - cameraPosition);
            float3 worldAxis = normalize(mul((float3x3)instance.normalMatrix, bounds.cone.xyz));
            float4 worldCone = float4(worldAxis, bounds.cone.w);
            
            if (!ConeCull(worldCone, cameraDir)) {
                visible = true;
            }
        }
    }
    
    // Count visible meshlets using wave intrinsics
    uint visibleMask = WaveActiveBallot(visible).x;
    uint visibleCount = WaveActiveCountBits(visible);
    uint laneIndex = WaveGetLaneIndex();
    
    // First lane writes count
    if (laneIndex == 0) {
        gs_visibleCount = visibleCount;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Compact visible meshlet indices
    if (visible) {
        uint writeIndex = WavePrefixCountBits(visible);
        gs_visibleMeshlets[writeIndex] = globalMeshletIndex;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Dispatch mesh shader groups for visible meshlets
    MeshPayload payload;
    payload.instanceId = instanceId;
    
    [unroll]
    for (uint i = 0; i < 32; ++i) {
        payload.meshletIndices[i] = (i < gs_visibleCount) ? gs_visibleMeshlets[i] : 0;
    }
    
    // Dispatch one mesh shader group per visible meshlet
    DispatchMesh(gs_visibleCount, 1, 1, payload);
}
