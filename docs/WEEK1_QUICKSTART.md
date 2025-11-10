# Week 1 Quick-Start Guide
## 🎯 Goal: Enhanced Buildings + Basic SSAO + Better Player Character

This guide provides concrete, copy-paste ready code to immediately enhance your cityscape.

---

## 📋 Priority Tasks

### ✅ Task 1: Add Window Lights to Buildings (30 mins)
**Impact**: HIGH - Instantly makes city feel alive

#### Step 1: Enhance BuildingVertex Structure
**File**: `src_refactored/Rendering/CudaBuildingKernels.cu`

Add emissive color support:
```cuda
struct BuildingVertex {
    float3 position;
    float3 normal;
    float2 uv;
    float3 color;
    float3 emissive;  // ADD THIS - for glowing windows
};
```

#### Step 2: Update Building Generation Kernel
Add window light logic after line 158:

```cuda
// After setting vertColor, add window light logic
float3 emissiveColor = float3(0, 0, 0);  // Default: no glow

// Only on vertical faces (walls)
if (tid >= 0 && tid <= 3) {
    // Generate window pattern based on UV coordinates
    int windowX = int(faceUVs[i].x * 4.0f);  // 4 windows wide
    int windowY = int(faceUVs[i].y * 8.0f);  // 8 floors tall
    
    // Hash to determine if window is lit
    uint32_t windowHash = style.seed + tid * 1000 + windowY * 100 + windowX;
    float isLit = hash(windowHash);
    
    // 60% of windows are lit
    if (isLit > 0.4f) {
        // Warm yellow-orange glow
        emissiveColor = float3(1.0f, 0.85f, 0.6f) * 2.0f;
    }
}

vertices[baseVert + i].emissive = emissiveColor;
```

#### Step 3: Update Shader to Use Emissive
**File**: `assets/shaders/geometry_pass.frag`

Add after line where you output color:
```glsl
// In fragment shader
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedo;  // RGB = color, A = emissive intensity

// ... in main() ...
float emissivePower = length(vEmissive);  // From vertex shader
gAlbedo = vec4(albedo, emissivePower);
```

**File**: `assets/shaders/lighting_pass.frag`

Blend emissive into final color:
```glsl
vec3 lighting = ambientLight + diffuseLight + specularLight;
vec3 emissive = texture(gAlbedo, TexCoords).rgb * texture(gAlbedo, TexCoords).a * 2.0;
FragColor = vec4(lighting + emissive, 1.0);
```

---

### ✅ Task 2: Add Architectural Details (45 mins)
**Impact**: MEDIUM - Adds visual complexity

#### Enhance Building Geometry with Trim
**File**: `src_refactored/Rendering/CudaBuildingKernels.cu`

Add new kernel for detail pass:
```cuda
__global__ void AddBuildingDetailsKernel(
    BuildingVertex* vertices,
    uint32_t* indices,
    const BuildingStyleGPU style,
    int baseVertexCount,
    int* outVertexCount,
    int* outIndexCount
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread adds one detail element
    if (tid >= 12) return;  // 12 detail elements
    
    const float hw = style.baseWidth * 0.5f;
    const float hd = style.baseDepth * 0.5f;
    const float h = style.height;
    
    int detailVert = baseVertexCount + tid * 4;
    int detailIdx = (baseVertexCount / 4 * 6) + tid * 6;
    
    // Detail types:
    // 0-3: Corner trim pieces
    // 4-7: Floor separator bands
    // 8-11: Roofline decorations
    
    if (tid < 4) {
        // Corner trim
        float trimWidth = 0.3f;
        float3 cornerPos;
        float3 cornerNormal;
        
        switch(tid) {
            case 0: cornerPos = float3(-hw, 0, hd); cornerNormal = float3(-0.707f, 0, 0.707f); break;
            case 1: cornerPos = float3(hw, 0, hd); cornerNormal = float3(0.707f, 0, 0.707f); break;
            case 2: cornerPos = float3(hw, 0, -hd); cornerNormal = float3(0.707f, 0, -0.707f); break;
            case 3: cornerPos = float3(-hw, 0, -hd); cornerNormal = float3(-0.707f, 0, -0.707f); break;
        }
        
        // Create vertical trim strip
        vertices[detailVert + 0].position = cornerPos;
        vertices[detailVert + 1].position = float3(cornerPos.x + trimWidth * cornerNormal.x, 
                                                    cornerPos.y, 
                                                    cornerPos.z + trimWidth * cornerNormal.z);
        vertices[detailVert + 2].position = float3(cornerPos.x + trimWidth * cornerNormal.x, 
                                                    h, 
                                                    cornerPos.z + trimWidth * cornerNormal.z);
        vertices[detailVert + 3].position = float3(cornerPos.x, h, cornerPos.z);
        
        for (int i = 0; i < 4; i++) {
            vertices[detailVert + i].normal = cornerNormal;
            vertices[detailVert + i].color = style.accentColor;  // Darker trim
            vertices[detailVert + i].emissive = float3(0, 0, 0);
        }
    }
    else if (tid >= 4 && tid < 8) {
        // Floor separator bands (every 3 units = 1 floor)
        int floorNum = tid - 4;  // 0-3
        float floorHeight = (floorNum + 1) * 3.0f;
        
        if (floorHeight < h) {
            float bandHeight = 0.2f;
            int faceIdx = tid - 4;  // Which wall face
            
            // Create horizontal band around building at floor level
            // ... (similar vertex generation for horizontal strip)
        }
    }
    
    // Write indices (two triangles)
    indices[detailIdx + 0] = detailVert + 0;
    indices[detailIdx + 1] = detailVert + 1;
    indices[detailIdx + 2] = detailVert + 2;
    indices[detailIdx + 3] = detailVert + 0;
    indices[detailIdx + 4] = detailVert + 2;
    indices[detailIdx + 5] = detailVert + 3;
    
    // Update counts
    if (tid == 0) {
        *outVertexCount = baseVertexCount + 48;  // 12 details * 4 verts
        *outIndexCount = (baseVertexCount / 4 * 6) + 72;  // base + (12 * 6)
    }
}
```

#### Call Detail Kernel from Generator
**File**: `src_refactored/Rendering/CudaBuildingGenerator.cpp`

In `GenerateBaseGeometry()`, add after main kernel:
```cpp
// Launch detail kernel
int detailBlockSize = 16;
int detailBlocks = 1;

int tempVertCount, tempIdxCount;
AddBuildingDetailsKernel<<<detailBlocks, detailBlockSize>>>(
    deviceVertices,
    deviceIndices,
    gpuStyle,
    24,  // Base vertex count
    &tempVertCount,
    &tempIdxCount
);

cudaDeviceSynchronize();
```

---

### ✅ Task 3: Implement Basic SSAO (60 mins)
**Impact**: VERY HIGH - Adds dramatic depth perception

#### Create SSAO Kernel File
**New File**: `src_refactored/Rendering/CudaSSAOKernels.cu`

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct SSAOParameters {
    int sampleCount;
    float radius;
    float bias;
    float power;  // Exponent for darkening
};

// Sample kernel (Poisson disk distribution)
__constant__ float3 sampleKernel[64];

// Random rotation vectors
__device__ inline float3 getRandomVector(int x, int y) {
    uint32_t hash = x * 73856093 ^ y * 19349663;
    float angle = float(hash % 360) * 0.017453f;  // degrees to radians
    return float3(cosf(angle), sinf(angle), 0.0f);
}

__global__ void ComputeSSAOKernel(
    const float* gBufferPosition,   // RGB = world position
    const float* gBufferNormal,     // RGB = world normal
    const float* gBufferDepth,      // R = depth
    float* ssaoOutput,              // R = occlusion factor (0-1)
    const SSAOParameters params,
    const float* viewMatrix,
    const float* projMatrix,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixelIdx = y * width + x;
    
    // Read G-buffer data
    float3 fragPos = make_float3(
        gBufferPosition[pixelIdx * 3 + 0],
        gBufferPosition[pixelIdx * 3 + 1],
        gBufferPosition[pixelIdx * 3 + 2]
    );
    
    float3 normal = make_float3(
        gBufferNormal[pixelIdx * 3 + 0],
        gBufferNormal[pixelIdx * 3 + 1],
        gBufferNormal[pixelIdx * 3 + 2]
    );
    
    float depth = gBufferDepth[pixelIdx];
    
    // Skip background (far plane)
    if (depth > 0.9999f) {
        ssaoOutput[pixelIdx] = 1.0f;  // No occlusion
        return;
    }
    
    // Random rotation vector
    float3 randomVec = getRandomVector(x, y);
    
    // TBN matrix for sample kernel rotation
    float3 tangent = normalize(cross(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    
    // Sample surrounding positions
    float occlusion = 0.0f;
    
    for (int i = 0; i < params.sampleCount; i++) {
        // Orient sample
        float3 sampleDir = sampleKernel[i];
        float3 samplePos = fragPos + 
                          tangent * sampleDir.x * params.radius +
                          bitangent * sampleDir.y * params.radius +
                          normal * sampleDir.z * params.radius;
        
        // Project sample position to screen space
        // ... (matrix multiply, perspective divide)
        // Compare sample depth with G-buffer depth
        // Accumulate occlusion
        
        // Simplified: if sample is closer than G-buffer, it's occluded
        occlusion += (samplePos.y < fragPos.y - params.bias) ? 1.0f : 0.0f;
    }
    
    occlusion = 1.0f - (occlusion / float(params.sampleCount));
    occlusion = powf(occlusion, params.power);  // Artistic control
    
    ssaoOutput[pixelIdx] = occlusion;
}

// Blur pass for smooth SSAO
__global__ void BlurSSAOKernel(
    const float* ssaoInput,
    float* ssaoOutput,
    int width,
    int height,
    int blurRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -blurRadius; dy <= blurRadius; dy++) {
        for (int dx = -blurRadius; dx <= blurRadius; dx++) {
            int sx = x + dx;
            int sy = y + dy;
            
            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                sum += ssaoInput[sy * width + sx];
                count++;
            }
        }
    }
    
    ssaoOutput[y * width + x] = sum / float(count);
}
```

#### Initialize Sample Kernel (CPU side)
**File**: `src_refactored/Rendering/CudaRenderingSystem.cpp`

In `InitializeCudaResources()`:
```cpp
// Generate SSAO sample kernel
std::vector<glm::vec3> ssaoKernel;
std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
std::default_random_engine generator;

for (int i = 0; i < 64; i++) {
    glm::vec3 sample(
        randomFloats(generator) * 2.0f - 1.0f,
        randomFloats(generator) * 2.0f - 1.0f,
        randomFloats(generator)  // Hemisphere
    );
    sample = glm::normalize(sample);
    sample *= randomFloats(generator);
    
    // Scale samples (more near origin)
    float scale = float(i) / 64.0f;
    scale = 0.1f + scale * scale * 0.9f;
    sample *= scale;
    
    ssaoKernel.push_back(sample);
}

// Copy to CUDA constant memory
cudaMemcpyToSymbol(sampleKernel, ssaoKernel.data(), 
                   64 * sizeof(glm::vec3), 0, cudaMemcpyHostToDevice);
```

---

### ✅ Task 4: Improve Player Character (30 mins)
**Impact**: MEDIUM - Clearer visual identity

#### Create Simple Low-Poly Humanoid
**New File**: `src_refactored/Rendering/ProceduralCharacter.cpp`

```cpp
#include "Rendering/Mesh.h"
#include <glm/glm.hpp>

namespace CudaGame {
namespace Rendering {

Mesh CreateLowPolyCharacter() {
    Mesh mesh;
    
    // Simple humanoid design (200 triangles)
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> colors;
    std::vector<uint32_t> indices;
    
    // Color scheme
    glm::vec3 bodyColor(0.2f, 0.4f, 0.8f);    // Blue jacket
    glm::vec3 pantsColor(0.1f, 0.1f, 0.2f);    // Dark pants
    glm::vec3 skinColor(0.9f, 0.7f, 0.6f);     // Skin tone
    glm::vec3 hairColor(0.2f, 0.15f, 0.1f);    // Brown hair
    
    // Body parts (simplified boxes)
    
    // 1. Head (cube, 0.5 units)
    float headSize = 0.5f;
    float headY = 1.7f;  // Height of head
    AddCube(positions, normals, colors, indices,
            glm::vec3(0, headY, 0), headSize, skinColor);
    
    // 2. Hair (flat box on top of head)
    AddBox(positions, normals, colors, indices,
           glm::vec3(0, headY + headSize * 0.6f, 0),
           glm::vec3(headSize * 1.1f, 0.2f, headSize * 1.1f),
           hairColor);
    
    // 3. Torso (box, 0.7 wide x 0.9 tall x 0.4 deep)
    AddBox(positions, normals, colors, indices,
           glm::vec3(0, 1.0f, 0),
           glm::vec3(0.7f, 0.9f, 0.4f),
           bodyColor);
    
    // 4. Arms (two thin boxes)
    float armWidth = 0.2f;
    float armLength = 0.7f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-0.5f, 1.0f, 0),  // Left arm
           glm::vec3(armWidth, armLength, armWidth),
           bodyColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(0.5f, 1.0f, 0),   // Right arm
           glm::vec3(armWidth, armLength, armWidth),
           bodyColor);
    
    // 5. Legs (two boxes)
    float legWidth = 0.25f;
    float legLength = 0.9f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-0.2f, 0.45f, 0),  // Left leg
           glm::vec3(legWidth, legLength, legWidth),
           pantsColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(0.2f, 0.45f, 0),   // Right leg
           glm::vec3(legWidth, legLength, legWidth),
           pantsColor);
    
    // 6. Simple eyes (two small cubes)
    float eyeSize = 0.08f;
    glm::vec3 eyeColor(0, 0, 0);  // Black
    AddCube(positions, normals, colors, indices,
            glm::vec3(-0.15f, headY + 0.1f, headSize * 0.5f),
            eyeSize, eyeColor);
    AddCube(positions, normals, colors, indices,
            glm::vec3(0.15f, headY + 0.1f, headSize * 0.5f),
            eyeSize, eyeColor);
    
    // Convert to mesh
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.colors = colors;
    mesh.indices = indices;
    
    return mesh;
}

// Helper function to add a cube
void AddCube(std::vector<glm::vec3>& positions,
             std::vector<glm::vec3>& normals,
             std::vector<glm::vec3>& colors,
             std::vector<uint32_t>& indices,
             glm::vec3 center,
             float size,
             glm::vec3 color) {
    AddBox(positions, normals, colors, indices,
           center, glm::vec3(size), color);
}

// Helper function to add a box
void AddBox(std::vector<glm::vec3>& positions,
            std::vector<glm::vec3>& normals,
            std::vector<glm::vec3>& colors,
            std::vector<uint32_t>& indices,
            glm::vec3 center,
            glm::vec3 size,
            glm::vec3 color) {
    // ... (generate 24 vertices for box, 36 indices)
    // Similar to building cube generation
}

} // namespace Rendering
} // namespace CudaGame
```

#### Replace Player Mesh in Game
**File**: Your main game initialization

```cpp
// Instead of creating debug cube:
auto characterMesh = Rendering::CreateLowPolyCharacter();
UploadMeshToGPU(characterMesh);  // Your existing upload function
```

---

## 🔧 Integration Checklist

### Before Starting:
- [ ] Backup current project to `OLD/` folder
- [ ] Ensure CUDA builds successfully (`cmake --build build --config Release`)
- [ ] Take screenshot of current state for comparison

### Implementation Order:
1. [ ] **Window lights** (quick win, immediate visual impact)
2. [ ] **Player character** (replace cube, clear identity)
3. [ ] **Architectural details** (adds depth, not critical path)
4. [ ] **SSAO** (big impact, but more complex integration)

### Testing After Each Feature:
```bash
# Rebuild
cmake --build build --config Release

# Run
cd build/Release
./Full3DGame.exe
```

### Expected Results:
- **Window lights**: Buildings glow at night, feel alive
- **Player character**: Clear humanoid shape, distinctive colors
- **Details**: Corner trim, floor bands visible
- **SSAO**: Dark shadows at building bases, between structures

---

## 🐛 Troubleshooting

### Windows Don't Glow:
- Check emissive values in shader
- Verify `gAlbedo.a` channel contains emissive power
- Ensure lighting pass blends emissive correctly

### SSAO Too Dark/Light:
Adjust parameters in `SSAOParameters`:
```cpp
params.sampleCount = 32;   // Lower = faster, noisier
params.radius = 1.5f;      // Larger = wider shadows
params.bias = 0.025f;      // Higher = less acne
params.power = 2.0f;       // Higher = more contrast
```

### Character Model Not Showing:
- Check mesh upload (VAO/VBO created?)
- Verify transform matrix
- Check camera frustum culling

---

## 📊 Performance Targets

### After Week 1 Implementation:
- **FPS**: Should maintain 60+ FPS
- **Frame time**: <16ms
- **Memory**: +100-200MB (SSAO buffers)

### Profiling Commands:
```cpp
// Add to your render loop
auto startTime = std::chrono::high_resolution_clock::now();

// ... render code ...

auto endTime = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
std::cout << "Frame time: " << duration.count() / 1000.0f << "ms" << std::endl;
```

---

## 🎨 Visual Comparison

### Before:
- Flat buildings, uniform color
- Debug cube player
- No shadows
- Feels empty

### After Week 1:
- Buildings with glowing windows
- Distinctive character model
- Architectural details (trim, bands)
- Contact shadows (SSAO)
- Scene feels alive and inhabited

---

## 🚀 Next: Week 2 Preview

Once Week 1 is complete, Week 2 will add:
- Volumetric fog
- Time-of-day lighting
- Cel-shading post-process
- Edge detection outlines

**These build on the G-buffer and SSAO foundation from Week 1!**

---

## 💬 Questions?

Common issues and solutions will be documented as you progress. The key is to implement features incrementally and test after each addition.

**Good luck! Your cityscape is about to look amazing! 🌆✨**
