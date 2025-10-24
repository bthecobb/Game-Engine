#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include <iomanip>
#include <unordered_map>
#include <atomic>
#include <condition_variable>
#include <deque>

namespace CudaGame {
namespace Core {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR_LEVEL = 4,  // Renamed from ERROR to avoid conflict with Windows macro
    CRITICAL = 5,
    FATAL = 6
};

enum class LogCategory {
    CORE,
    RENDERING,
    PHYSICS,
    GAMEPLAY,
    AUDIO,
    NETWORKING,
    AI,
    ANIMATION,
    PARTICLES,
    INPUT,
    MEMORY,
    PERFORMANCE,
    ASSET,
    SCRIPT,
    UI
};

struct LogEntry {
    LogLevel level;
    LogCategory category;
    std::string message;
    std::string file;
    int line;
    std::string function;
    std::chrono::system_clock::time_point timestamp;
    std::thread::id threadId;
    
    LogEntry(LogLevel lvl, LogCategory cat, const std::string& msg, 
             const std::string& f, int l, const std::string& func)
        : level(lvl), category(cat), message(msg), file(f), line(l), 
          function(func), timestamp(std::chrono::system_clock::now()),
          threadId(std::this_thread::get_id()) {}
};

class Logger {
public:
    static Logger& GetInstance() {
        static Logger instance;
        return instance;
    }
    
    // Initialize logging system
    bool Initialize(const std::string& logDirectory = "logs/");
    void Shutdown();
    
    // Core logging functions
    void Log(LogLevel level, LogCategory category, const std::string& message,
             const std::string& file, int line, const std::string& function);
    
    // Set minimum log level
    void SetLogLevel(LogLevel level) { m_minLogLevel = level; }
    void SetCategoryFilter(LogCategory category, bool enabled);
    
    // Performance logging
    void LogFrameTime(float deltaTime);
    void LogMemoryUsage(size_t bytes, const std::string& category);
    void LogGPUEvent(const std::string& event, float milliseconds);
    
    // Flush logs to file
    void Flush();
    
    // Get statistics
    struct LogStats {
        std::unordered_map<LogLevel, size_t> levelCounts;
        std::unordered_map<LogCategory, size_t> categoryCounts;
        size_t totalLogs = 0;
        float avgFrameTime = 0.0f;
        size_t peakMemoryUsage = 0;
    };
    LogStats GetStats() const { return m_stats; }
    
    // Console colors for different log levels
    void EnableColoredOutput(bool enable) { m_useColors = enable; }
    
    // Performance profiling
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& name, LogCategory category);
        ~ScopedTimer();
    private:
        std::string m_name;
        LogCategory m_category;
        std::chrono::high_resolution_clock::time_point m_start;
    };
    
private:
    Logger();
    ~Logger();
    
    void ProcessLogQueue();
    void WriteToFile(const LogEntry& entry);
    void WriteToConsole(const LogEntry& entry);
    std::string FormatLogEntry(const LogEntry& entry);
    std::string GetLogLevelString(LogLevel level) const;
    std::string GetCategoryString(LogCategory category) const;
    std::string GetColorCode(LogLevel level) const;
    
    // File management
    std::ofstream m_logFile;
    std::ofstream m_perfLogFile;
    std::string m_logDirectory;
    
    // Threading
    std::thread m_logThread;
    std::queue<LogEntry> m_logQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::atomic<bool> m_running;
    
    // Settings
    LogLevel m_minLogLevel = LogLevel::DEBUG;
    std::unordered_map<LogCategory, bool> m_categoryFilters;
    bool m_useColors = true;
    bool m_logToFile = true;
    bool m_logToConsole = true;
    
    // Statistics
    mutable LogStats m_stats;
    mutable std::mutex m_statsMutex;
    
    // Frame timing
    std::deque<float> m_frameTimes;
    const size_t MAX_FRAME_SAMPLES = 120;
};

// Convenience macros for logging
#define LOG_TRACE(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::TRACE, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_DEBUG(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::DEBUG, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_INFO(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::INFO, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_WARNING(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::WARNING, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_ERROR(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::ERROR_LEVEL, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_CRITICAL(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::CRITICAL, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

#define LOG_FATAL(category, message) \
    CudaGame::Core::Logger::GetInstance().Log(CudaGame::Core::LogLevel::FATAL, \
    CudaGame::Core::LogCategory::category, message, __FILE__, __LINE__, __FUNCTION__)

// Performance logging macro
#define LOG_SCOPE_TIMER(name, category) \
    CudaGame::Core::Logger::ScopedTimer _timer##__LINE__(name, CudaGame::Core::LogCategory::category)

// Memory logging macro
#define LOG_MEMORY(bytes, category) \
    CudaGame::Core::Logger::GetInstance().LogMemoryUsage(bytes, category)

// GPU event logging macro
#define LOG_GPU_EVENT(event, ms) \
    CudaGame::Core::Logger::GetInstance().LogGPUEvent(event, ms)

} // namespace Core
} // namespace CudaGame
