#include "Testing/TestFramework.h"
#include <iostream>
#include "Testing/TestDebugger.h"
// Forward declarations for test suite creation functions
std::shared_ptr<CudaGame::Testing::TestSuite> CreateCoreSystemsTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateOrbitCameraTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateSkyboxUtilsTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateBuildingGeneratorTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateEnemyAITestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateThirdPersonCameraRigTestSuite();
#ifdef ENABLE_PHYSX
std::shared_ptr<CudaGame::Testing::TestSuite> CreateCharacterControllerTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreatePhysicsTestSuite();
#endif
#ifdef ENABLE_RENDER_TESTS
// std::shared_ptr<CudaGame::Testing::TestSuite> CreateRenderingSystemTestSuite();
#endif
// std::shared_ptr<CudaGame::Testing::TestSuite> CreateFull3DGameIntegrationTestSuite();
// std::shared_ptr<CudaGame::Testing::TestSuite> CreatePlayerMovementTestSuite();
// std::shared_ptr<CudaGame::Testing::TestSuite> CreateEmissiveTextureTestSuite();

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "  AAA Game Engine Test Runner" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Enable verbose debug logging for diagnostics
    CudaGame::Testing::TestDebugger::SetVerbose(true);

    auto& testFramework = CudaGame::Testing::TestFramework::GetInstance();

    // Register all test suites
    testFramework.RegisterSuite(CreateCoreSystemsTestSuite());
    testFramework.RegisterSuite(CreateOrbitCameraTestSuite());
    testFramework.RegisterSuite(CreateSkyboxUtilsTestSuite());
    testFramework.RegisterSuite(CreateBuildingGeneratorTestSuite());
    testFramework.RegisterSuite(CreateEnemyAITestSuite());
    // Re-enable ThirdPersonCameraRig tests (consolidated in OrbitCameraTests.cpp)
    testFramework.RegisterSuite(CreateThirdPersonCameraRigTestSuite());
#ifdef ENABLE_PHYSX
    testFramework.RegisterSuite(CreateCharacterControllerTestSuite());
    // testFramework.RegisterSuite(CreatePhysicsTestSuite());
#endif
#ifdef ENABLE_RENDER_TESTS
    testFramework.RegisterSuite(CreateRenderingSystemTestSuite());
#endif
// testFramework.RegisterSuite(CreateEmissiveTextureTestSuite());
    // testFramework.RegisterSuite(CreateFull3DGameIntegrationTestSuite());
    // testFramework.RegisterSuite(CreatePlayerMovementTestSuite());

    // Run all tests
    testFramework.RunAllTests();

    // Exit with a non-zero code if any tests failed
    if (testFramework.GetFailedTests() > 0) {
        return 1;
    }

    return 0;
}
