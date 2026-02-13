#include "Testing/TestDebugger.h"
#include <iostream>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace CudaGame {
namespace Testing {

// Static member initialization
std::vector<TestDebugger::PerformanceRecord> TestDebugger::s_performanceHistory;
bool TestDebugger::s_verbose = false;
std::string TestDebugger::s_currentTest;
std::chrono::high_resolution_clock::time_point TestDebugger::s_testStartTime;

std::string TestDebugger::DumpEntityState(Core::Entity entity, Core::Coordinator& coord) {
    std::stringstream ss;
    ss << "\n=== Entity " << entity << " State ===\n";
    
    auto* entityMgr = coord.GetEntityManager();
    if (!entityMgr->IsEntityAlive(entity)) {
        ss << "Entity is DEAD/DESTROYED\n";
        return ss.str();
    }
    
    ss << "Entity is ALIVE\n";
    ss << "Signature: " << entityMgr->GetSignature(entity) << "\n";
    ss << "Components: " << GetEntityComponentList(entity, coord) << "\n";
    
    return ss.str();
}

std::string TestDebugger::GetEntityComponentList(Core::Entity entity, Core::Coordinator& coord) {
    std::stringstream ss;
    auto signature = coord.GetEntityManager()->GetSignature(entity);
    
    ss << "[";
    bool first = true;
    for (size_t i = 0; i < signature.size(); ++i) {
        if (signature.test(i)) {
            if (!first) ss << ", ";
            ss << "Component#" << i;
            first = false;
        }
    }
    if (first) {
        ss << "NONE";
    }
    ss << "]";
    
    return ss.str();
}

std::string TestDebugger::GetAllEntitiesInfo(Core::Coordinator& coord) {
    std::stringstream ss;
    auto* entityMgr = coord.GetEntityManager();
    
    ss << "\n=== All Entities Info ===\n";
    ss << "Living entities: " << entityMgr->GetLivingEntityCount() << "\n";
    
    int aliveCount = 0;
    for (Core::Entity e = 0; e < Core::MAX_ENTITIES && aliveCount < 100; ++e) {
        if (entityMgr->IsEntityAlive(e)) {
            ss << "Entity " << e << ": " << GetEntityComponentList(e, coord) << "\n";
            aliveCount++;
        }
    }
    
    if (aliveCount >= 100) {
        ss << "... (showing first 100 entities)\n";
    }
    
    return ss.str();
}

std::string TestDebugger::GetSystemEntityCounts(Core::Coordinator& coord) {
    std::stringstream ss;
    ss << "\n=== System Entity Counts ===\n";
    
    // Note: This is a simplified version
    // In a real implementation, you'd iterate through registered systems
    ss << "Entity manager total: " << coord.GetEntityManager()->GetLivingEntityCount() << "\n";
    
    return ss.str();
}

std::string TestDebugger::GetSystemInfo(Core::Coordinator& coord) {
    std::stringstream ss;
    ss << "\n=== System Info ===\n";
    ss << "System manager active systems: " << coord.GetSystemManager()->GetSystemCount() << "\n";
    
    return ss.str();
}

std::string TestDebugger::GetComponentArrayStats(Core::Coordinator& coord) {
    std::stringstream ss;
    ss << "\n=== Component Array Statistics ===\n";
    ss << "Note: Detailed component array inspection requires template specialization\n";
    ss << "Entity count: " << coord.GetEntityManager()->GetLivingEntityCount() << "\n";
    
    return ss.str();
}

void TestDebugger::LogTestStart(const std::string& testName) {
    s_currentTest = testName;
    s_testStartTime = std::chrono::high_resolution_clock::now();
    
    if (s_verbose) {
        std::cout << "\n[TEST START] " << testName << " at " << GetTimestamp() << "\n";
    }
}

void TestDebugger::LogTestEnd(const std::string& testName, bool passed, float durationMs) {
    if (s_verbose) {
        std::cout << "[TEST END] " << testName 
                  << " - " << (passed ? "PASSED" : "FAILED")
                  << " (" << FormatDuration(durationMs) << ")\n";
    }
    
    RecordTestPerformance(testName, durationMs, passed);
}

void TestDebugger::SaveFailureContext(const std::string& testName, 
                                      const std::string& error,
                                      Core::Coordinator& coord) {
    std::stringstream filename;
    filename << "test_failure_" << testName << "_" << GetTimestamp() << ".txt";
    
    std::ofstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << "Failed to save failure context to " << filename.str() << std::endl;
        return;
    }
    
    file << "=================================\n";
    file << "Test Failure Context\n";
    file << "=================================\n";
    file << "Test: " << testName << "\n";
    file << "Timestamp: " << GetTimestamp() << "\n";
    file << "Error: " << error << "\n\n";
    
    file << GetAllEntitiesInfo(coord);
    file << GetSystemInfo(coord);
    file << GetEntityManagerState(coord);
    file << GetComponentArrayStats(coord);
    
    file.close();
    
    std::cout << "\n[DEBUG] Failure context saved to: " << filename.str() << "\n";
}

void TestDebugger::RecordTestPerformance(const std::string& testName, float durationMs, bool passed) {
    PerformanceRecord record;
    record.testName = testName;
    record.durationMs = durationMs;
    record.passed = passed;
    record.timestamp = std::chrono::system_clock::now();
    
    s_performanceHistory.push_back(record);
}

std::string TestDebugger::GetPerformanceReport() {
    std::stringstream ss;
    ss << "\n=== Performance Report ===\n";
    ss << "Total tests run: " << s_performanceHistory.size() << "\n\n";
    
    // Calculate statistics
    float totalTime = 0.0f;
    float minTime = std::numeric_limits<float>::max();
    float maxTime = 0.0f;
    int passedCount = 0;
    
    std::map<std::string, std::vector<float>> testTimes;
    
    for (const auto& record : s_performanceHistory) {
        totalTime += record.durationMs;
        if (record.durationMs < minTime) minTime = record.durationMs;
        if (record.durationMs > maxTime) maxTime = record.durationMs;
        if (record.passed) passedCount++;
        
        testTimes[record.testName].push_back(record.durationMs);
    }
    
    ss << "Total execution time: " << FormatDuration(totalTime) << "\n";
    ss << "Fastest test: " << FormatDuration(minTime) << "\n";
    ss << "Slowest test: " << FormatDuration(maxTime) << "\n";
    ss << "Pass rate: " << passedCount << "/" << s_performanceHistory.size() 
       << " (" << (100.0f * passedCount / s_performanceHistory.size()) << "%)\n\n";
    
    // Per-test averages
    ss << "Per-test averages:\n";
    for (const auto& pair : testTimes) {
        float avg = 0.0f;
        for (float time : pair.second) {
            avg += time;
        }
        avg /= pair.second.size();
        ss << "  " << std::setw(40) << std::left << pair.first 
           << ": " << FormatDuration(avg) 
           << " (runs: " << pair.second.size() << ")\n";
    }
    
    return ss.str();
}

std::vector<TestDebugger::PerformanceRecord> TestDebugger::GetPerformanceHistory() {
    return s_performanceHistory;
}

void TestDebugger::CheckMemoryLeaks() {
    // Basic memory leak detection
    std::cout << "\n[MEMORY CHECK] Current usage: " 
              << (GetCurrentMemoryUsage() / 1024 / 1024) << " MB\n";
}

size_t TestDebugger::GetCurrentMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

std::string TestDebugger::GetEntityManagerState(Core::Coordinator& coord) {
    std::stringstream ss;
    auto* entityMgr = coord.GetEntityManager();
    
    ss << "\n=== Entity Manager State ===\n";
    ss << "Living entities: " << entityMgr->GetLivingEntityCount() << "\n";
    ss << "Max entities: " << Core::MAX_ENTITIES << "\n";
    ss << "Utilization: " << (100.0f * entityMgr->GetLivingEntityCount() / Core::MAX_ENTITIES) << "%\n";
    
    return ss.str();
}

void TestDebugger::ExportDiagnostics(const std::string& filename, Core::Coordinator& coord) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to export diagnostics to " << filename << std::endl;
        return;
    }
    
    file << "=================================\n";
    file << "Test Diagnostics Export\n";
    file << "=================================\n";
    file << "Timestamp: " << GetTimestamp() << "\n\n";
    
    file << GetEntityManagerState(coord);
    file << GetSystemInfo(coord);
    file << GetAllEntitiesInfo(coord);
    file << GetPerformanceReport();
    
    file.close();
    std::cout << "[DEBUG] Diagnostics exported to: " << filename << "\n";
}

std::string TestDebugger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    char buffer[100];
#ifdef _WIN32
    struct tm timeinfo;
    localtime_s(&timeinfo, &time);
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &timeinfo);
#else
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", localtime(&time));
#endif
    
    return std::string(buffer);
}

std::string TestDebugger::FormatDuration(float ms) {
    std::stringstream ss;
    if (ms < 1.0f) {
        ss << std::fixed << std::setprecision(3) << (ms * 1000.0f) << " μs";
    } else if (ms < 1000.0f) {
        ss << std::fixed << std::setprecision(2) << ms << " ms";
    } else {
        ss << std::fixed << std::setprecision(2) << (ms / 1000.0f) << " s";
    }
    return ss.str();
}

} // namespace Testing
} // namespace CudaGame
