#include "Testing/TestFramework.h"
#include "Rendering/CudaBuildingGenerator.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <memory>
#include <cmath>

using namespace CudaGame::Testing;
using namespace CudaGame::Rendering;

std::shared_ptr<TestSuite> CreateBuildingGeneratorTestSuite()
{
    auto suite = std::make_shared<TestSuite>("Building Generator");
    
    // Test 1: Verify roof geometry exists
    suite->AddTest("Generates Roof Geometry", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        BuildingStyle style;
        style.baseWidth = 10.0f;
        style.baseDepth = 10.0f;
        style.height = 15.0f;
        
        BuildingMesh mesh = generator->GenerateBuilding(style);
        
        // Should have at least 24 vertices (6 faces * 4 vertices)
        ASSERT_GE(mesh.positions.size(), 24);
        
        // Verify roof vertices exist at the top
        bool hasTopFace = false;
        int topVertexCount = 0;
        
        for (size_t i = 0; i < mesh.positions.size(); ++i) {
            const auto& pos = mesh.positions[i];
            const auto& normal = mesh.normals[i];
            
            // Check if this is a top face vertex
            if (std::abs(pos.y - style.height) < 0.01f && normal.y > 0.9f) {
                hasTopFace = true;
                topVertexCount++;
            }
        }
        
        ASSERT_TRUE(hasTopFace);
        ASSERT_GE(topVertexCount, 4);
        
        generator->Shutdown();
    });
    
    // Test 2: Verify normals point upward
    suite->AddTest("Roof Normals Point Upward", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        BuildingStyle style;
        style.baseWidth = 8.0f;
        style.baseDepth = 8.0f;
        style.height = 12.0f;
        
        BuildingMesh mesh = generator->GenerateBuilding(style);
        
        // Find all top vertices and verify their normals
        for (size_t i = 0; i < mesh.positions.size(); ++i) {
            const auto& pos = mesh.positions[i];
            const auto& normal = mesh.normals[i];
            
            // If this is a top vertex (at height)
            if (std::abs(pos.y - style.height) < 0.01f) {
                // Normal should point upward (0, 1, 0)
                ASSERT_NEAR(normal.x, 0.0f, 0.01f);
                ASSERT_NEAR(normal.y, 1.0f, 0.01f);
                ASSERT_NEAR(normal.z, 0.0f, 0.01f);
            }
        }
        
        generator->Shutdown();
    });
    
    // Test 3: Verify roof covers full area
    suite->AddTest("Roof Covers Full Area", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        BuildingStyle style;
        style.baseWidth = 10.0f;
        style.baseDepth = 12.0f;
        style.height = 20.0f;
        
        BuildingMesh mesh = generator->GenerateBuilding(style);
        
        // Find all top vertices
        std::vector<glm::vec3> topVertices;
        for (size_t i = 0; i < mesh.positions.size(); ++i) {
            if (std::abs(mesh.positions[i].y - style.height) < 0.01f &&
                mesh.normals[i].y > 0.9f) {
                topVertices.push_back(mesh.positions[i]);
            }
        }
        
        ASSERT_GE(topVertices.size(), 4);
        
        // Calculate bounds of top face
        glm::vec3 minPos(FLT_MAX);
        glm::vec3 maxPos(-FLT_MAX);
        
        for (const auto& v : topVertices) {
            minPos = glm::min(minPos, v);
            maxPos = glm::max(maxPos, v);
        }
        
        // Roof should span the full width and depth
        float roofWidth = maxPos.x - minPos.x;
        float roofDepth = maxPos.z - minPos.z;
        
        ASSERT_NEAR(roofWidth, style.baseWidth, 0.1f);
        ASSERT_NEAR(roofDepth, style.baseDepth, 0.1f);
        
        generator->Shutdown();
    });
    
    // Test 4: Verify bounding box includes roof
    suite->AddTest("Bounding Box Includes Roof", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        BuildingStyle style;
        style.baseWidth = 8.0f;
        style.baseDepth = 6.0f;
        style.height = 18.0f;
        
        BuildingMesh mesh = generator->GenerateBuilding(style);
        
        // Bounding box should extend to full height
        ASSERT_GE(mesh.boundsMax.y, style.height - 0.01f);
        ASSERT_LE(mesh.boundsMin.y, 0.01f);
        
        generator->Shutdown();
    });
    
    // Test 5: Test different building sizes
    suite->AddTest("Different Sizes Generate Correct Roofs", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        std::vector<BuildingStyle> styles = {
            BuildingStyle{BuildingStyle::Type::MODERN, 6.0f, 6.0f, 10.0f},
            BuildingStyle{BuildingStyle::Type::OFFICE, 12.0f, 8.0f, 20.0f},
            BuildingStyle{BuildingStyle::Type::SKYSCRAPER, 10.0f, 10.0f, 30.0f}
        };
        
        for (const auto& style : styles) {
            BuildingMesh mesh = generator->GenerateBuilding(style);
            
            // Count top vertices
            int topCount = 0;
            for (size_t i = 0; i < mesh.positions.size(); ++i) {
                if (std::abs(mesh.positions[i].y - style.height) < 0.01f &&
                    mesh.normals[i].y > 0.9f) {
                    topCount++;
                }
            }
            
            ASSERT_GE(topCount, 4);
        }
        
        generator->Shutdown();
    });
    
    // Test 6: Verify triangle winding order
    suite->AddTest("Roof Triangle Winding Order Correct", []() {
        auto generator = std::make_unique<CudaBuildingGenerator>();
        ASSERT_TRUE(generator->Initialize());
        
        BuildingStyle style;
        style.baseWidth = 10.0f;
        style.baseDepth = 10.0f;
        style.height = 15.0f;
        
        BuildingMesh mesh = generator->GenerateBuilding(style);
        
        // Find top face vertices
        std::vector<size_t> topVertexIndices;
        for (size_t i = 0; i < mesh.positions.size(); ++i) {
            if (std::abs(mesh.positions[i].y - style.height) < 0.01f &&
                mesh.normals[i].y > 0.9f) {
                topVertexIndices.push_back(i);
            }
        }
        
        // Verify indices reference top vertices and have correct winding
        bool foundTopTriangle = false;
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            uint32_t i0 = mesh.indices[i];
            uint32_t i1 = mesh.indices[i + 1];
            uint32_t i2 = mesh.indices[i + 2];
            
            // Check if all three vertices are top vertices
            bool allTop = std::find(topVertexIndices.begin(), topVertexIndices.end(), i0) != topVertexIndices.end() &&
                          std::find(topVertexIndices.begin(), topVertexIndices.end(), i1) != topVertexIndices.end() &&
                          std::find(topVertexIndices.begin(), topVertexIndices.end(), i2) != topVertexIndices.end();
            
            if (allTop) {
                foundTopTriangle = true;
                
                // Verify triangle winding order is correct (CCW when viewed from above)
                glm::vec3 v0 = mesh.positions[i0];
                glm::vec3 v1 = mesh.positions[i1];
                glm::vec3 v2 = mesh.positions[i2];
                
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));
                
                // Face normal should point upward for top face
                ASSERT_GT(faceNormal.y, 0.9f);
            }
        }
        
        ASSERT_TRUE(foundTopTriangle);
        
        generator->Shutdown();
    });
    
    return suite;
}
