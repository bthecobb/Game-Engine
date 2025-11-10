#include "Testing/TestFramework.h"
#include "Rendering/SkyboxUtils.h"
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <memory>
#include <cmath>

using namespace CudaGame::Testing;
using namespace CudaGame::Rendering;

static float AngleBetween(const glm::vec3& a, const glm::vec3& b) {
    float d = glm::clamp(glm::dot(glm::normalize(a), glm::normalize(b)), -1.0f, 1.0f);
    return std::acos(d);
}

std::shared_ptr<TestSuite> CreateSkyboxUtilsTestSuite()
{
    auto suite = std::make_shared<TestSuite>("SkyboxUtilsTests");

    suite->AddTest("Equirect_RoundTrip_Directions", [](){
        glm::vec3 dirs[] = {
            {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
            glm::normalize(glm::vec3(1,1,0)), glm::normalize(glm::vec3(1,0,1)), glm::normalize(glm::vec3(1,1,1))
        };
        for (const auto& d : dirs) {
            glm::vec2 uv = DirectionToEquirectUV(d);
            glm::vec3 back = EquirectUVToDirection(uv);
            float ang = AngleBetween(d, back);
            ASSERT_LT(ang, 1e-3f);
        }
    });

    suite->AddTest("Cubemap_Face_Selection_CanonicalAxes", [](){
        CubemapSample px = DirectionToCubemap({1,0,0});
        CubemapSample nx = DirectionToCubemap({-1,0,0});
        CubemapSample py = DirectionToCubemap({0,1,0});
        CubemapSample ny = DirectionToCubemap({0,-1,0});
        CubemapSample pz = DirectionToCubemap({0,0,1});
        CubemapSample nz = DirectionToCubemap({0,0,-1});
        ASSERT_EQ(px.faceIndex, 0);
        ASSERT_EQ(nx.faceIndex, 1);
        ASSERT_EQ(py.faceIndex, 2);
        ASSERT_EQ(ny.faceIndex, 3);
        ASSERT_EQ(pz.faceIndex, 4);
        ASSERT_EQ(nz.faceIndex, 5);
        // UV for canonical axes should be at center (~0.5, 0.5)
        auto near = [](float a, float b){ return std::abs(a-b) < 1e-5f; };
        ASSERT_TRUE(near(px.uv.x, 0.5f) && near(px.uv.y, 0.5f));
        ASSERT_TRUE(near(nx.uv.x, 0.5f) && near(nx.uv.y, 0.5f));
        ASSERT_TRUE(near(py.uv.x, 0.5f) && near(py.uv.y, 0.5f));
        ASSERT_TRUE(near(ny.uv.x, 0.5f) && near(ny.uv.y, 0.5f));
        ASSERT_TRUE(near(pz.uv.x, 0.5f) && near(pz.uv.y, 0.5f));
        ASSERT_TRUE(near(nz.uv.x, 0.5f) && near(nz.uv.y, 0.5f));
    });

    suite->AddTest("MipLevel_Calculation_CommonSizes", [](){
        ASSERT_EQ(ComputeMipLevels(1), 1);
        ASSERT_EQ(ComputeMipLevels(2), 2);
        ASSERT_EQ(ComputeMipLevels(4), 3);
        ASSERT_EQ(ComputeMipLevels(8), 4);
        ASSERT_EQ(ComputeMipLevels(512), 10);
        ASSERT_EQ(ComputeMipLevels(1024), 11);
        ASSERT_EQ(ComputeMipLevels(300), 9); // floor(log2(300)) + 1 = 8 + 1
    });

    suite->AddTest("Gamma_Conversions_Idempotence_Endpoints", [](){
        glm::vec3 black(0.0f);
        glm::vec3 white(1.0f);
        ASSERT_TRUE(glm::all(glm::epsilonEqual(LinearToSRGB(black), black, 1e-6f)));
        ASSERT_TRUE(glm::all(glm::epsilonEqual(LinearToSRGB(white), white, 1e-6f)));
        ASSERT_TRUE(glm::all(glm::epsilonEqual(SRGBToLinear(black), black, 1e-6f)));
        ASSERT_TRUE(glm::all(glm::epsilonEqual(SRGBToLinear(white), white, 1e-6f)));
    });

    return suite;
}
