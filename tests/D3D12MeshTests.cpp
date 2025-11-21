#ifdef _WIN32
#include <gtest/gtest.h>
#include "Rendering/D3D12Mesh.h"
#include "Rendering/Backends/DX12RenderBackend.h"
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

using namespace CudaGame::Rendering;

class D3D12MeshTest : public ::testing::Test {
protected:
    void SetUp() override {
        backend = std::make_unique<DX12RenderBackend>();
        ASSERT_TRUE(backend->Initialize());
    }

    void TearDown() override {
        backend.reset();
    }

    std::unique_ptr<DX12RenderBackend> backend;
};

// Test 1: Create cube mesh
TEST_F(D3D12MeshTest, CreateCube) {
    auto cube = MeshGenerator::CreateCube(backend.get());
    ASSERT_NE(cube, nullptr);
    
    // Cube should have 24 vertices (4 per face, 6 faces)
    EXPECT_EQ(cube->GetVertexCount(), 24u);
    
    // Cube should have 36 indices (2 triangles per face, 6 faces)
    EXPECT_EQ(cube->GetIndexCount(), 36u);
    
    // Check material defaults
    const auto& mat = cube->GetMaterial();
    EXPECT_EQ(mat.albedoColor, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    EXPECT_EQ(mat.roughness, 0.5f);
    EXPECT_EQ(mat.metallic, 0.0f);
}

// Test 2: Create sphere mesh
TEST_F(D3D12MeshTest, CreateSphere) {
    auto sphere = MeshGenerator::CreateSphere(backend.get(), 16);
    ASSERT_NE(sphere, nullptr);
    
    // Sphere with 16 segments should have specific vertex/index count
    // (rings+1) * (sectors+1) where sectors = segments * 2
    uint32_t expectedVerts = (16 + 1) * (32 + 1);
    EXPECT_EQ(sphere->GetVertexCount(), expectedVerts);
    
    // Should have indices
    EXPECT_GT(sphere->GetIndexCount(), 0u);
}

// Test 3: Create plane mesh
TEST_F(D3D12MeshTest, CreatePlane) {
    auto plane = MeshGenerator::CreatePlane(backend.get());
    ASSERT_NE(plane, nullptr);
    
    // Plane is a simple quad
    EXPECT_EQ(plane->GetVertexCount(), 4u);
    EXPECT_EQ(plane->GetIndexCount(), 6u);
}

// Test 4: Mesh material modification
TEST_F(D3D12MeshTest, MaterialModification) {
    auto cube = MeshGenerator::CreateCube(backend.get());
    ASSERT_NE(cube, nullptr);
    
    // Modify material
    auto& mat = cube->GetMaterial();
    mat.albedoColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red
    mat.roughness = 0.8f;
    mat.metallic = 0.3f;
    
    // Verify changes
    const auto& matConst = cube->GetMaterial();
    EXPECT_EQ(matConst.albedoColor.r, 1.0f);
    EXPECT_EQ(matConst.albedoColor.g, 0.0f);
    EXPECT_EQ(matConst.roughness, 0.8f);
    EXPECT_EQ(matConst.metallic, 0.3f);
}

// Test 5: Mesh transform
TEST_F(D3D12MeshTest, MeshTransform) {
    auto cube = MeshGenerator::CreateCube(backend.get());
    ASSERT_NE(cube, nullptr);
    
    // Set transform
    cube->transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    cube->transform = glm::rotate(cube->transform, glm::radians(45.0f), glm::vec3(0, 1, 0));
    cube->transform = glm::scale(cube->transform, glm::vec3(2.0f, 2.0f, 2.0f));
    
    // Just verify it doesn't crash
    EXPECT_NE(cube->transform[0][0], 0.0f);
}

// Test 6: Multiple meshes
TEST_F(D3D12MeshTest, MultipleMeshes) {
    auto cube = MeshGenerator::CreateCube(backend.get());
    auto sphere = MeshGenerator::CreateSphere(backend.get());
    auto plane = MeshGenerator::CreatePlane(backend.get());
    
    EXPECT_NE(cube, nullptr);
    EXPECT_NE(sphere, nullptr);
    EXPECT_NE(plane, nullptr);
    
    // All should be independent
    EXPECT_NE(cube->GetIndexCount(), sphere->GetIndexCount());
    EXPECT_NE(cube->GetIndexCount(), plane->GetIndexCount());
}

// Test 7: High-res sphere
TEST_F(D3D12MeshTest, HighResSphere) {
    auto sphere = MeshGenerator::CreateSphere(backend.get(), 64);
    ASSERT_NE(sphere, nullptr);
    
    // High-res sphere should have many vertices
    EXPECT_GT(sphere->GetVertexCount(), 1000u);
    EXPECT_GT(sphere->GetIndexCount(), 3000u);
}

#else
// Non-Windows platform - provide empty tests
#include <gtest/gtest.h>

TEST(D3D12MeshTest, NotAvailableOnThisPlatform) {
    GTEST_SKIP() << "D3D12 mesh system only available on Windows";
}
#endif // _WIN32
