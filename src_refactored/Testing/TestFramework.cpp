#include "Testing/TestFramework.h"
#include <iostream>
#include <iomanip>

namespace CudaGame {
namespace Testing {

TestSuite::TestSuite(const std::string& suiteName) : m_suiteName(suiteName) {}

void TestSuite::AddTest(const std::string& testName, std::function<void()> testFunction) {
    m_tests.emplace_back(testName, testFunction);
}

TestResult TestSuite::RunTest(const std::string& testName) {
    TestResult result;
    result.testName = testName;

    try {
        auto testIt = std::find_if(m_tests.begin(), m_tests.end(), 
                                   [&](const auto& t) { return t.first == testName; });

        if (testIt != m_tests.end()) {
            auto start = std::chrono::high_resolution_clock::now();
            testIt->second();
            auto end = std::chrono::high_resolution_clock::now();
            result.executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.passed = true;
        } else {
            result.errorMessage = "Test not found in suite.";
        }
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
    }

    return result;
}

std::vector<TestResult> TestSuite::RunAllTests() {
    std::vector<TestResult> results;
    for (const auto& test : m_tests) {
        results.push_back(RunTest(test.first));
    }
    return results;
}

TestFramework& TestFramework::GetInstance() {
    static TestFramework instance;
    return instance;
}

void TestFramework::RegisterSuite(std::shared_ptr<TestSuite> suite) {
    m_testSuites.push_back(suite);
}

void TestFramework::RunAllTests() {
    m_allResults.clear();
    for (const auto& suite : m_testSuites) {
        std::cout << "\n--- Running Test Suite: " << suite->GetSuiteName() << " ---\n" << std::endl;
        auto suiteResults = suite->RunAllTests();
        for (const auto& result : suiteResults) {
            std::cout << "[TEST] " << std::setw(50) << std::left << result.testName 
                      << (result.passed ? "[  PASSED  ]" : "[!!FAILED!!]") << std::endl;
            if (!result.passed) {
                std::cout << "       ERROR: " << result.errorMessage << std::endl;
            }
            m_allResults.push_back(result);
        }
    }
    PrintResults();
}

void TestFramework::RunTestSuite(const std::string& suiteName) {
    for (const auto& suite : m_testSuites) {
        if (suite->GetSuiteName() == suiteName) {
            suite->RunAllTests();
            break;
        }
    }
}

void TestFramework::PrintResults() const {
    size_t passed = GetPassedTests();
    size_t failed = GetFailedTests();
    size_t total = GetTotalTests();

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Total Tests: " << total << std::endl;
    std::cout << "Passed:      " << passed << " ("
              << std::fixed << std::setprecision(2) << (total > 0 ? (100.0 * passed / total) : 0.0) << "%)" << std::endl;
    std::cout << "Failed:      " << failed << std::endl;
    std::cout << "------------------" << std::endl;
}

size_t TestFramework::GetTotalTests() const {
    return m_allResults.size();
}

size_t TestFramework::GetPassedTests() const {
    size_t count = 0;
    for (const auto& result : m_allResults) {
        if (result.passed) {
            count++;
        }
    }
    return count;
}

size_t TestFramework::GetFailedTests() const {
    return GetTotalTests() - GetPassedTests();
}

} // namespace Testing
} // namespace CudaGame
