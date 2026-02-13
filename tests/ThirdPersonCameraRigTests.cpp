#include "Testing/TestFramework.h"
#include "Rendering/OrbitCamera.h"
#include "Rendering/ThirdPersonCameraRig.h"
#include <glm/glm.hpp>
#include <memory>

using namespace CudaGame::Testing;
using namespace CudaGame::Rendering;

namespace {
struct RigFixture {
    std::unique_ptr<OrbitCamera> cam;
    ThirdPersonCameraRig rig;

    void SetUp() {
        // Reset rig to a clean state per test
        rig = ThirdPersonCameraRig{};
        cam = std::make_unique<OrbitCamera>(ProjectionType::PERSPECTIVE);
        OrbitCamera::OrbitSettings s; s.distance = 4.5f; s.heightOffset = 0.0f; s.smoothSpeed = 12.0f;
        cam->SetOrbitSettings(s);
        cam->SetPerspective(60.0f, 16.0f/9.0f, 0.1f, 200.0f);
        ThirdPersonCameraRig::Settings rs; rs.distance=4.5f; rs.height=1.6f; rs.shoulderOffsetX=0.45f; rs.followSmooth=18.0f;
        rig.SetCamera(cam.get()); rig.Configure(rs);
        // seed camera target
        cam->SetTarget(glm::vec3(0.0f, rs.height, 0.0f));
    }

    void TearDown() { cam.reset(); }
};
}

// Legacy rig test suite retained for reference; superseded by consolidated suite in OrbitCameraTests.cpp
std::shared_ptr<TestSuite> CreateThirdPersonCameraRigTestSuite_Legacy() {
    auto suite = std::make_shared<TestSuite>("ThirdPerson Camera Rig (Legacy)");
    auto fixture = std::make_shared<RigFixture>();

    suite->AddTest("Rig settles to new target without overshoot", [fixture]() {
        fixture->SetUp();
        glm::vec3 posA(0.0f, 2.0f, 0.0f), posB(5.0f, 2.0f, 0.0f);
        glm::vec3 vel(0.0f);
        // jump to new player position
        for (int i=0;i<90;++i) {
            fixture->rig.Update(1.0f/60.0f, posB, vel);
        }
        glm::vec3 tgt = fixture->cam->GetTarget();
        ASSERT_NEAR(tgt.x, 5.0f, 0.2f);
        ASSERT_NEAR(tgt.z, 0.0f, 0.2f);
        fixture->TearDown();
    });

    suite->AddTest("Strafe flip stability (no oscillation spikes)", [fixture]() {
        fixture->SetUp();
        glm::vec3 left(-1.0f, 2.0f, 0.0f), right(1.0f, 2.0f, 0.0f);
        glm::vec3 vel(0.0f);
        float maxStep = 0.0f;
        float lastX = fixture->cam->GetTarget().x;
        for (int i=0;i<120;++i) {
            glm::vec3 p = (i%2==0)? left : right;
            fixture->rig.Update(1.0f/60.0f, p, vel);
            float x = fixture->cam->GetTarget().x;
            maxStep = std::max(maxStep, std::abs(x - lastX));
            lastX = x;
        }
        // step should remain bounded due to 2nd-order smoothing and not spike wildly
        ASSERT_LT(maxStep, 1.0f);
        fixture->TearDown();
    });

    return suite;
}
