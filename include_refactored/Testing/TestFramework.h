#pragma once

#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <iostream>

namespace CudaGame {
namespace Testing {

// Test result structure
struct TestResult {
    std::string testName;
    bool passed = false;
    std::string errorMessage;
    std::chrono::microseconds executionTime{0};
};

// Test suite class
class TestSuite {
public:
    TestSuite(const std::string& suiteName);
    ~TestSuite() = default;

    // Add test cases
    void AddTest(const std::string& testName, std::function<void()> testFunction);
    
    // Run all tests in the suite
    std::vector<TestResult> RunAllTests();
    
    // Run a specific test
    TestResult RunTest(const std::string& testName);
    
    // Get suite statistics
    size_t GetTestCount() const { return m_tests.size(); }
    const std::string& GetSuiteName() const { return m_suiteName; }

private:
    std::string m_suiteName;
    std::vector<std::pair<std::string, std::function<void()>>> m_tests;
};

// Test framework singleton
class TestFramework {
public:
    static TestFramework& GetInstance();
    
    // Register test suites
    void RegisterSuite(std::shared_ptr<TestSuite> suite);
    
    // Run all test suites
    void RunAllTests();
    
    // Run a specific test suite
    void RunTestSuite(const std::string& suiteName);
    
    // Print test results
    void PrintResults() const;
    
    // Get overall statistics
    size_t GetTotalTests() const;
    size_t GetPassedTests() const;
    size_t GetFailedTests() const;

private:
    TestFramework() = default;
    std::vector<std::shared_ptr<TestSuite>> m_testSuites;
    std::vector<TestResult> m_allResults;
};

// Assertion macros
#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        throw std::runtime_error("Assertion failed: " #condition " should be false"); \
    }

#define ASSERT_EQ(expected, actual) \
    if ((expected) != (actual)) { \
        throw std::runtime_error("Assertion failed: " #expected " != " #actual); \
    }

#define ASSERT_NE(expected, actual) \
    if ((expected) == (actual)) { \
        throw std::runtime_error("Assertion failed: values should not be equal"); \
    }

#define ASSERT_NEAR(expected, actual, tolerance) \
    if (std::abs((expected) - (actual)) > (tolerance)) { \
        throw std::runtime_error("Assertion failed: values not within tolerance"); \
    }

#define ASSERT_LT(a, b) \
    if (!((a) < (b))) { \
        throw std::runtime_error("Assertion failed: " #a " < " #b); \
    }

#define ASSERT_GT(a, b) \
    if (!((a) > (b))) { \
        throw std::runtime_error("Assertion failed: " #a " > " #b); \
    }

#define ASSERT_LE(a, b) \
    if (!((a) <= (b))) { \
        throw std::runtime_error("Assertion failed: " #a " <= " #b); \
    }

#define ASSERT_GE(a, b) \
    if (!((a) >= (b))) { \
        throw std::runtime_error("Assertion failed: " #a " >= " #b); \
    }

#define ASSERT_NULL(ptr) \
    if ((ptr) != nullptr) { \
        throw std::runtime_error("Assertion failed: pointer should be null"); \
    }

#define ASSERT_NOT_NULL(ptr) \
    if ((ptr) == nullptr) { \
        throw std::runtime_error("Assertion failed: pointer should not be null"); \
    }

// Performance testing macros
#define BENCHMARK_START() \
    auto benchmark_start = std::chrono::high_resolution_clock::now();

#define BENCHMARK_END(name) \
    auto benchmark_end = std::chrono::high_resolution_clock::now(); \
    auto benchmark_duration = std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start); \
    std::cout << "[BENCHMARK] " << name << ": " << benchmark_duration.count() << " microseconds" << std::endl;

} // namespace Testing
} // namespace CudaGame
