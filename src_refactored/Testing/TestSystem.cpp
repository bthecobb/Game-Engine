#include "Testing/TestSystem.h"
#include <filesystem>

namespace CudaGame {
namespace Testing {

TestSystem& TestSystem::GetInstance() {
    static TestSystem instance;
    return instance;
}

void TestSystem::RegisterScenario(const std::string& name, float duration, std::function<void()> setup) {
    // In a real system, we might store these in a map to select by usage.
    // For now, we will add them to the queue when StartSuite is called if we had a list.
    // But simplistic approach: RegisterScenario ADDS to the queue immediately? 
    // No, usually we register available scenarios, then BuildQueue.
    
    // Simplification: We assume Register is called during Setup, and we queue them.
    // But if we want to run "all", we need to store them.
    // Let's change RegisterScenario to just Store. Use StartSuite to Queue.
    // Actually, for this iteration, let's assume Register = Add to potential list.
    // BUT since we call Register in `main` or `init`, we need storage.
    
    // Let's modify logic inside StartSuite to populate.
    // Wait, the API `RegisterScenario` implies defining it.
    // I need a member `std::vector<TestScenario> m_registeredScenarios;`
    // I missed that in the header. I will add it here as a static or member if I can modify header.
    // I'll implementation-define a static list here if the header didn't have it?
    // No, header defines class layout.
    // I'll stick to a simpler flow: "Register" queues it immediately for now, assuming we validly want to run it.
    // Or I'll rely on the user calling Register BEFORE StartSuite.
    
    TestScenario s;
    s.name = name;
    s.duration = duration;
    s.setup = setup;
    m_testQueue.push(s);
}

void TestSystem::StartSuite(const std::string& suiteName) {
    (void)suiteName; // Currently runs everything in queue
    if (m_testQueue.empty()) {
        std::cout << "[TestSystem] No scenarios registered." << std::endl;
        return;
    }
    
    std::cout << "[TestSystem] Starting Test Suite." << std::endl;
    m_isActive = true;
    StartNextTest();
    
    // Create output directory
    std::filesystem::create_directories("TestResults");
}

void TestSystem::Update(float deltaTime) {
    if (!m_isActive) return;
    
    m_currentTestTime += deltaTime;
    m_frameCounter++;
    
    if (m_currentTestTime >= m_currentTest.duration) {
        StartNextTest();
    }
}

void TestSystem::StartNextTest() {
    if (m_testQueue.empty()) {
        FinishSuite();
        return;
    }
    
    m_currentTest = m_testQueue.front();
    m_testQueue.pop();
    
    m_currentTestTime = 0.0f;
    m_frameCounter = 0;
    
    std::cout << "[TestSystem] Running Scenario: " << m_currentTest.name << std::endl;
    
    // Create subdir
    std::filesystem::create_directories("TestResults/" + m_currentTest.name);
    
    if (m_currentTest.setup) {
        m_currentTest.setup();
    }
}

void TestSystem::FinishSuite() {
    std::cout << "[TestSystem] Suite Complete. Exiting." << std::endl;
    m_isActive = false;
    exit(0); // Clean exit after tests
}

bool TestSystem::ShouldCaptureFrame() const {
    // Capture every 5th frame to save disk IO, or every frame for smoothness
    return m_isActive && (m_frameCounter % 5 == 0);
}

}
}
