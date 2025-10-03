#include "Testing/TestFramework.h"
#include <iostream>

// Forward declarations for test suite creation functions
std::shared_ptr<CudaGame::Testing::TestSuite> CreateCoreSystemsTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateOrbitCameraTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateCharacterControllerTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateRenderingSystemTestSuite();
std::shared_ptr<CudaGame::Testing::TestSuite> CreateFull3DGameIntegrationTestSuite();
// std::shared_ptr<CudaGame::Testing::TestSuite> CreatePhysicsTestSuite();

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "  AAA Game Engine Test Runner" << std::endl;
    std::cout << "=================================" << std::endl;

    auto& testFramework = CudaGame::Testing::TestFramework::GetInstance();

    // Register all test suites
    testFramework.RegisterSuite(CreateCoreSystemsTestSuite());
    testFramework.RegisterSuite(CreateOrbitCameraTestSuite());
    testFramework.RegisterSuite(CreateCharacterControllerTestSuite());
    testFramework.RegisterSuite(CreateRenderingSystemTestSuite());
    testFramework.RegisterSuite(CreateFull3DGameIntegrationTestSuite());
    // testFramework.RegisterSuite(CreatePhysicsTestSuite());

    // Run all tests
    testFramework.RunAllTests();

    // Exit with a non-zero code if any tests failed
    if (testFramework.GetFailedTests() > 0) {
        return 1;
    }

    return 0;
}
