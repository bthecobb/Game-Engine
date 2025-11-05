#include "Testing/TestFramework.h"
#include "Rendering/CudaBuildingGenerator.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <cmath>

using namespace CudaGame::Testing;
using namespace CudaGame::Rendering;

static inline int idx(int x, int y, int w) { return (y * w + x) * 4; }

static void sampleRGBA(const std::vector<uint8_t>& data, int w, int h, float u, float v,
                       float& r, float& g, float& b, float& a)
{
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    int x = (int)std::round(u * (w - 1));
    int y = (int)std::round(v * (h - 1));
    int i = idx(x, y, w);
    r = data[i+0] / 255.0f;
    g = data[i+1] / 255.0f;
    b = data[i+2] / 255.0f;
    a = data[i+3] / 255.0f;
}

std::shared_ptr<TestSuite> CreateEmissiveTextureTestSuite()
{
    auto suite = std::make_shared<TestSuite>("EmissiveTextureTests");

    suite->AddTest("EmissiveTexture_Generation_Basic", [](){
        CudaBuildingGenerator gen;
        ASSERT_TRUE(gen.Initialize());

        BuildingStyle style;
        style.height = 24.0f; // ~8 floors
        style.seed = 1337;

        auto tex = gen.GenerateBuildingTexture(style, 512);
        ASSERT_EQ((int)tex.emissiveData.size(), tex.width * tex.height * 4);
        // Expect some lit pixels
        size_t lit = 0;
        for (size_t i = 3; i < tex.emissiveData.size(); i += 4) if (tex.emissiveData[i] > 0) ++lit;
        ASSERT_GT((int)lit, tex.width * tex.height / 50); // >2% pixels lit

        gen.Shutdown();
    });

    suite->AddTest("EmissiveTexture_WindowGrid_RatioAndZones", [](){
        CudaBuildingGenerator gen; ASSERT_TRUE(gen.Initialize());
        BuildingStyle style; style.height = 30.0f; style.seed = 42;
        auto tex = gen.GenerateBuildingTexture(style, 512);

        const int W = tex.width, H = tex.height;
        const float windowsWide = 6.0f, floorsHigh = 10.0f; // must match kernel
        const float margin = 0.15f;

        int windows = 0, litWindows = 0, wallSamples = 0, wallLit = 0;
        for (int wy = 0; wy < (int)floorsHigh; ++wy) {
            for (int wx = 0; wx < (int)windowsWide; ++wx) {
                float u = (wx + 0.5f) / windowsWide;
                float v = (wy + 0.5f) / floorsHigh;
                float r,g,b,a; sampleRGBA(tex.emissiveData, W, H, u, v, r,g,b,a);
                ++windows; if (a > 0.01f) ++litWindows;

                // Sample a wall region inside same cell near left margin
                float uw = (wx + margin * 0.5f) / windowsWide;
                float vw = (wy + 0.5f) / floorsHigh;
                sampleRGBA(tex.emissiveData, W, H, uw, vw, r,g,b,a);
                ++wallSamples; if (a > 0.01f) ++wallLit;
            }
        }
        // Expect ~70% lit with tolerance [50%, 90%]
        float litRatio = (float)litWindows / (float)windows;
        ASSERT_GE(litRatio, 0.50f); ASSERT_LE(litRatio, 0.90f);
        // Walls should be dark
        ASSERT_LE((float)wallLit / (float)wallSamples, 0.02f);
        gen.Shutdown();
    });

    suite->AddTest("EmissiveTexture_ColorPallette_Validation", [](){
        CudaBuildingGenerator gen; ASSERT_TRUE(gen.Initialize());
        BuildingStyle style; style.height = 30.0f; style.seed = 7;
        auto tex = gen.GenerateBuildingTexture(style, 512);

        const float colors[4][3] = {
            {1.0f, 0.9f, 0.7f}, // warm
            {0.5f, 0.7f, 1.0f}, // blue
            {1.0f, 0.4f, 0.9f}, // pink
            {0.6f, 1.0f, 0.6f}  // green
        };
        const float tol = 0.15f;

        int checked = 0; int ok = 0;
        for (int y = 0; y < tex.height; y += 64) {
            for (int x = 0; x < tex.width; x += 64) {
                float r = tex.emissiveData[idx(x,y,tex.width)+0] / 255.0f;
                float g = tex.emissiveData[idx(x,y,tex.width)+1] / 255.0f;
                float b = tex.emissiveData[idx(x,y,tex.width)+2] / 255.0f;
                float a = tex.emissiveData[idx(x,y,tex.width)+3] / 255.0f;
                if (a < 0.01f) continue; // only check lit
                ++checked;
                for (int i = 0; i < 4; ++i) {
                    if (std::fabs(r - colors[i][0]) <= tol &&
                        std::fabs(g - colors[i][1]) <= tol &&
                        std::fabs(b - colors[i][2]) <= tol) { ++ok; break; }
                }
            }
        }
        ASSERT_GT(ok, 0);
        ASSERT_GE((float)ok / std::max(1,checked), 0.5f);
        gen.Shutdown();
    });

    suite->AddTest("EmissiveTexture_Seed_Reproducibility", [](){
        CudaBuildingGenerator gen; ASSERT_TRUE(gen.Initialize());
        BuildingStyle style; style.height = 30.0f; style.seed = 999;
        auto a = gen.GenerateBuildingTexture(style, 256);
        auto b = gen.GenerateBuildingTexture(style, 256);
        ASSERT_EQ(a.emissiveData.size(), b.emissiveData.size());
        ASSERT_TRUE(std::equal(a.emissiveData.begin(), a.emissiveData.end(), b.emissiveData.begin()));
        style.seed = 1000; auto c = gen.GenerateBuildingTexture(style, 256);
        ASSERT_FALSE(std::equal(a.emissiveData.begin(), a.emissiveData.end(), c.emissiveData.begin()));
        gen.Shutdown();
    });

    return suite;
}
