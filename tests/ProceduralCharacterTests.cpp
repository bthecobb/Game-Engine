#include "gtest/gtest.h"
#include "Rendering/ProceduralCharacter.h"

using namespace CudaGame::Rendering;

TEST(ProceduralCharacter, CpuMeshBasicProperties) {
    Mesh m = CreateLowPolyCharacter();
    ASSERT_GT(m.vertices.size(), 0u);
    ASSERT_GT(m.indices.size(), 0u);

    float minY = 1e9f, maxY = -1e9f;
    for (const auto& v : m.vertices) {
        minY = std::min(minY, v.Position.y);
        maxY = std::max(maxY, v.Position.y);
    }
    float height = maxY - minY;
    // Expect roughly 2.0 units tall (+/- 20%)
    EXPECT_GT(height, 1.6f);
    EXPECT_LT(height, 2.4f);
}