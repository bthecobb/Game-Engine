#include "Rendering/Mesh.h"
#include <glm/glm.hpp>
#include <vector>

namespace CudaGame {
namespace Rendering {

// Helper to add a box
void AddBox(std::vector<glm::vec3>& positions,
            std::vector<glm::vec3>& normals,
            std::vector<glm::vec3>& colors,
            std::vector<uint32_t>& indices,
            glm::vec3 center,
            glm::vec3 size,
            glm::vec3 color) {
    
    int baseIdx = positions.size();
    
    float hw = size.x * 0.5f;
    float hh = size.y * 0.5f;
    float hd = size.z * 0.5f;
    
    // 24 vertices (6 faces * 4 vertices)
    glm::vec3 vertices[24] = {
        // Front face (+Z)
        {-hw, -hh, hd}, {hw, -hh, hd}, {hw, hh, hd}, {-hw, hh, hd},
        // Back face (-Z)
        {hw, -hh, -hd}, {-hw, -hh, -hd}, {-hw, hh, -hd}, {hw, hh, -hd},
        // Right face (+X)
        {hw, -hh, hd}, {hw, -hh, -hd}, {hw, hh, -hd}, {hw, hh, hd},
        // Left face (-X)
        {-hw, -hh, -hd}, {-hw, -hh, hd}, {-hw, hh, hd}, {-hw, hh, -hd},
        // Top face (+Y)
        {-hw, hh, -hd}, {hw, hh, -hd}, {hw, hh, hd}, {-hw, hh, hd},
        // Bottom face (-Y)
        {-hw, -hh, hd}, {hw, -hh, hd}, {hw, -hh, -hd}, {-hw, -hh, -hd}
    };
    
    glm::vec3 faceNormals[6] = {
        {0, 0, 1}, {0, 0, -1}, {1, 0, 0},
        {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}
    };
    
    // Add vertices
    for (int face = 0; face < 6; face++) {
        for (int v = 0; v < 4; v++) {
            positions.push_back(center + vertices[face * 4 + v]);
            normals.push_back(faceNormals[face]);
            colors.push_back(color);
        }
    }
    
    // Add indices (2 triangles per face)
    for (int face = 0; face < 6; face++) {
        int base = baseIdx + face * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

// Create a low-poly humanoid character
Mesh CreateLowPolyCharacter() {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> colors;
    std::vector<uint32_t> indices;
    
    // Color scheme - vibrant for visibility
    glm::vec3 jacketColor(0.2f, 0.5f, 0.95f);   // Bright blue jacket
    glm::vec3 pantsColor(0.15f, 0.15f, 0.25f);  // Dark pants
    glm::vec3 skinColor(0.95f, 0.75f, 0.65f);   // Skin tone
    glm::vec3 hairColor(0.25f, 0.18f, 0.12f);   // Brown hair
    glm::vec3 shoeColor(0.1f, 0.1f, 0.1f);      // Black shoes
    glm::vec3 eyeColor(0.0f, 0.0f, 0.0f);       // Black eyes
    
    // Body proportions (scaled for 2.0 unit tall character)
    float headSize = 0.4f;
    float headY = 1.65f;
    
    // 1. Head
    AddBox(positions, normals, colors, indices,
           glm::vec3(0, headY, 0),
           glm::vec3(headSize, headSize, headSize),
           skinColor);
    
    // 2. Hair (flat box on top)
    AddBox(positions, normals, colors, indices,
           glm::vec3(0, headY + headSize * 0.55f, 0),
           glm::vec3(headSize * 1.1f, 0.15f, headSize * 1.1f),
           hairColor);
    
    // 3. Eyes (two small cubes)
    float eyeSize = 0.08f;
    float eyeOffset = headSize * 0.3f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-eyeOffset, headY + 0.08f, headSize * 0.52f),
           glm::vec3(eyeSize, eyeSize, eyeSize * 0.5f),
           eyeColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(eyeOffset, headY + 0.08f, headSize * 0.52f),
           glm::vec3(eyeSize, eyeSize, eyeSize * 0.5f),
           eyeColor);
    
    // 4. Torso
    AddBox(positions, normals, colors, indices,
           glm::vec3(0, 1.05f, 0),
           glm::vec3(0.6f, 0.8f, 0.35f),
           jacketColor);
    
    // 5. Arms
    float armWidth = 0.18f;
    float armLength = 0.65f;
    float armY = 1.2f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-0.45f, armY, 0),  // Left arm
           glm::vec3(armWidth, armLength, armWidth),
           jacketColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(0.45f, armY, 0),   // Right arm
           glm::vec3(armWidth, armLength, armWidth),
           jacketColor);
    
    // 6. Legs
    float legWidth = 0.22f;
    float legLength = 0.85f;
    float legY = 0.425f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-0.18f, legY, 0),  // Left leg
           glm::vec3(legWidth, legLength, legWidth),
           pantsColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(0.18f, legY, 0),   // Right leg
           glm::vec3(legWidth, legLength, legWidth),
           pantsColor);
    
    // 7. Shoes
    float shoeHeight = 0.12f;
    AddBox(positions, normals, colors, indices,
           glm::vec3(-0.18f, shoeHeight * 0.5f, 0.08f),  // Left shoe
           glm::vec3(legWidth, shoeHeight, legWidth * 1.5f),
           shoeColor);
    AddBox(positions, normals, colors, indices,
           glm::vec3(0.18f, shoeHeight * 0.5f, 0.08f),   // Right shoe
           glm::vec3(legWidth, shoeHeight, legWidth * 1.5f),
           shoeColor);
    
    // Convert to Vertex format
    std::vector<Vertex> vertices;
    vertices.reserve(positions.size());
    
    for (size_t i = 0; i < positions.size(); i++) {
        Vertex v;
        v.Position = positions[i];
        v.Normal = normals[i];
        v.TexCoords = glm::vec2(0, 0);  // No UVs needed
        v.Tangent = glm::vec3(1, 0, 0);
        v.Bitangent = glm::vec3(0, 1, 0);
        vertices.push_back(v);
    }
    
    // Note: Color data is lost here, will need shader uniform or vertex attribute extension
    // For now, we'll use a uniform color in shader
    
    return Mesh(vertices, indices, {});  // Empty texture vector
}

} // namespace Rendering
} // namespace CudaGame
