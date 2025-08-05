#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>

namespace CudaGame {
namespace Profiling {

// Performance measurement data
struct ProfileData {
    std::string name;
    std::chrono::microseconds totalTime{0};
    std::chrono::microseconds minTime{std::chrono::microseconds::max()};
    std::chrono::microseconds maxTime{0};
    std::chrono::microseconds avgTime{0};
    size_t callCount = 0;
    bool isActive = false;
    std::chrono::high_resolution_clock::time_point startTime;
};

// RAII profile scope for automatic timing
class ProfileScope {
public:
    ProfileScope(const std::string& name);
    ~ProfileScope();

private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_startTime;
};

// Main profiler class
class Profiler {
public:
    static Profiler& GetInstance();

    // Profile management
    void StartProfile(const std::string& name);
    void EndProfile(const std::string& name);
    void ResetProfile(const std::string& name);
    void ResetAllProfiles();

    // Data access
    const ProfileData* GetProfileData(const std::string& name) const;
    std::vector<const ProfileData*> GetAllProfileData() const;

    // Reporting
    void PrintReport() const;
    void PrintFrameReport() const;
    void SaveReportToFile(const std::string& filename) const;

    // Frame-based profiling
    void BeginFrame();
    void EndFrame();
    double GetFPS() const { return m_currentFPS; }
    std::chrono::microseconds GetFrameTime() const { return m_lastFrameTime; }

    // System-specific profiling
    void ProfileCPUUsage();
    void ProfileMemoryUsage();
    void ProfileGPUUsage();

private:
    Profiler() = default;
    std::unordered_map<std::string, std::unique_ptr<ProfileData>> m_profiles;
    
    // Frame timing
    std::chrono::high_resolution_clock::time_point m_frameStartTime;
    std::chrono::microseconds m_lastFrameTime{0};
    double m_currentFPS = 0.0;
    std::vector<std::chrono::microseconds> m_recentFrameTimes;
    static constexpr size_t MAX_FRAME_HISTORY = 60;

    // System resources
    double m_cpuUsage = 0.0;
    size_t m_memoryUsage = 0;
    double m_gpuUsage = 0.0;
};

// GPU profiler for CUDA operations
class GPUProfiler {
public:
    static GPUProfiler& GetInstance();

    void StartGPUProfile(const std::string& name);
    void EndGPUProfile(const std::string& name);
    
    float GetGPUTime(const std::string& name) const;
    void PrintGPUReport() const;

private:
    GPUProfiler() = default;
    std::unordered_map<std::string, std::pair<void*, void*>> m_cudaEvents; // start, stop events
    std::unordered_map<std::string, float> m_gpuTimes;
};

// Convenience macros for profiling
#define PROFILE_SCOPE(name) CudaGame::Profiling::ProfileScope profile_scope_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#define PROFILE_BEGIN(name) CudaGame::Profiling::Profiler::GetInstance().StartProfile(name)
#define PROFILE_END(name) CudaGame::Profiling::Profiler::GetInstance().EndProfile(name)

#define GPU_PROFILE_BEGIN(name) CudaGame::Profiling::GPUProfiler::GetInstance().StartGPUProfile(name)
#define GPU_PROFILE_END(name) CudaGame::Profiling::GPUProfiler::GetInstance().EndGPUProfile(name)

} // namespace Profiling
} // namespace CudaGame
