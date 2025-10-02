#include "Core/Logger.h"
#include <filesystem>
#include <condition_variable>
#include <deque>

namespace CudaGame {
namespace Core {

Logger::Logger() : m_running(false) {}

Logger::~Logger() {
    if (m_running) {
        Shutdown();
    }
}

bool Logger::Initialize(const std::string& logDirectory) {
    m_logDirectory = logDirectory;
    
    // Create log directory if it doesn't exist
    std::filesystem::create_directories(m_logDirectory);
    
    // Generate timestamp for log filename
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << m_logDirectory << "/game_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
    
    // Open main log file
    m_logFile.open(ss.str(), std::ios::out | std::ios::app);
    if (!m_logFile.is_open()) {
        std::cerr << "Failed to open log file: " << ss.str() << std::endl;
        return false;
    }
    
    // Open performance log file
    ss.str("");
    ss << m_logDirectory << "/perf_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
    m_perfLogFile.open(ss.str(), std::ios::out | std::ios::app);
    
    // Initialize category filters (all enabled by default)
    for (int i = 0; i < 15; ++i) {
        m_categoryFilters[static_cast<LogCategory>(i)] = true;
    }
    
    // Start logging thread
    m_running = true;
    m_logThread = std::thread(&Logger::ProcessLogQueue, this);
    
    // Log initialization
    Log(LogLevel::INFO, LogCategory::CORE, "Logger initialized", __FILE__, __LINE__, __FUNCTION__);
    
    return true;
}

void Logger::Shutdown() {
    if (!m_running) return;
    
    Log(LogLevel::INFO, LogCategory::CORE, "Logger shutting down", __FILE__, __LINE__, __FUNCTION__);
    
    // Stop logging thread
    m_running = false;
    m_queueCV.notify_all();
    if (m_logThread.joinable()) {
        m_logThread.join();
    }
    
    // Close log files
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
    if (m_perfLogFile.is_open()) {
        m_perfLogFile.close();
    }
}

void Logger::Log(LogLevel level, LogCategory category, const std::string& message,
                 const std::string& file, int line, const std::string& function) {
    // Check if this log level and category should be logged
    if (level < m_minLogLevel) return;
    if (m_categoryFilters.find(category) != m_categoryFilters.end() && !m_categoryFilters[category]) return;
    
    // Create log entry
    LogEntry entry(level, category, message, file, line, function);
    
    // Add to queue
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_logQueue.push(entry);
        
        // Update statistics
        m_stats.levelCounts[level]++;
        m_stats.categoryCounts[category]++;
        m_stats.totalLogs++;
    }
    
    // Notify logging thread
    m_queueCV.notify_one();
}

void Logger::SetCategoryFilter(LogCategory category, bool enabled) {
    m_categoryFilters[category] = enabled;
}

void Logger::LogFrameTime(float deltaTime) {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    m_frameTimes.push_back(deltaTime);
    if (m_frameTimes.size() > MAX_FRAME_SAMPLES) {
        m_frameTimes.pop_front();
    }
    
    // Calculate average frame time
    float sum = 0.0f;
    for (float ft : m_frameTimes) {
        sum += ft;
    }
    m_stats.avgFrameTime = sum / m_frameTimes.size();
}

void Logger::LogMemoryUsage(size_t bytes, const std::string& category) {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    if (bytes > m_stats.peakMemoryUsage) {
        m_stats.peakMemoryUsage = bytes;
    }
    
    // Log memory event
    std::stringstream ss;
    ss << "Memory: " << category << " - " << (bytes / 1024.0 / 1024.0) << " MB";
    Log(LogLevel::DEBUG, LogCategory::MEMORY, ss.str(), __FILE__, __LINE__, __FUNCTION__);
}

void Logger::LogGPUEvent(const std::string& event, float milliseconds) {
    std::stringstream ss;
    ss << "GPU Event: " << event << " - " << milliseconds << " ms";
    Log(LogLevel::DEBUG, LogCategory::PERFORMANCE, ss.str(), __FILE__, __LINE__, __FUNCTION__);
}

void Logger::Flush() {
    // Wait for queue to be empty
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_queueCV.wait(lock, [this] { return m_logQueue.empty(); });
    
    // Flush files
    if (m_logFile.is_open()) {
        m_logFile.flush();
    }
    if (m_perfLogFile.is_open()) {
        m_perfLogFile.flush();
    }
}

void Logger::ProcessLogQueue() {
    while (m_running) {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        
        // Wait for logs or shutdown
        m_queueCV.wait(lock, [this] { return !m_logQueue.empty() || !m_running; });
        
        // Process all pending logs
        while (!m_logQueue.empty()) {
            LogEntry entry = m_logQueue.front();
            m_logQueue.pop();
            lock.unlock();
            
            // Write to file and console
            WriteToFile(entry);
            WriteToConsole(entry);
            
            lock.lock();
        }
    }
    
    // Process any remaining logs after shutdown
    std::lock_guard<std::mutex> lock(m_queueMutex);
    while (!m_logQueue.empty()) {
        LogEntry entry = m_logQueue.front();
        m_logQueue.pop();
        WriteToFile(entry);
        WriteToConsole(entry);
    }
}

void Logger::WriteToFile(const LogEntry& entry) {
    if (!m_logToFile || !m_logFile.is_open()) return;
    
    m_logFile << FormatLogEntry(entry) << std::endl;
}

void Logger::WriteToConsole(const LogEntry& entry) {
    if (!m_logToConsole) return;
    
    // Add color codes for console output
    if (m_useColors) {
        std::cout << GetColorCode(entry.level);
    }
    
    // Format and output
    std::cout << FormatLogEntry(entry);
    
    // Reset color
    if (m_useColors) {
        std::cout << "\033[0m";
    }
    
    std::cout << std::endl;
}

std::string Logger::FormatLogEntry(const LogEntry& entry) {
    std::stringstream ss;
    
    // Timestamp
    auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;
    
    ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    
    // Level and category
    ss << "[" << GetLogLevelString(entry.level) << "] ";
    ss << "[" << GetCategoryString(entry.category) << "] ";
    
    // Thread ID
    ss << "[T:" << entry.threadId << "] ";
    
    // Message
    ss << entry.message;
    
    // File info for debug and higher severity
    if (entry.level >= LogLevel::DEBUG) {
        // Extract just the filename from the full path
        std::string filename = entry.file;
        size_t lastSlash = filename.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            filename = filename.substr(lastSlash + 1);
        }
        
        ss << " (" << filename << ":" << entry.line << " in " << entry.function << ")";
    }
    
    return ss.str();
}

std::string Logger::GetLogLevelString(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE:    return "TRACE";
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO ";
        case LogLevel::WARNING:  return "WARN ";
        case LogLevel::ERROR_LEVEL:    return "ERROR";
        case LogLevel::CRITICAL: return "CRIT ";
        case LogLevel::FATAL:    return "FATAL";
        default:                 return "UNKN ";
    }
}

std::string Logger::GetCategoryString(LogCategory category) const {
    switch (category) {
        case LogCategory::CORE:        return "CORE     ";
        case LogCategory::RENDERING:   return "RENDER   ";
        case LogCategory::PHYSICS:     return "PHYSICS  ";
        case LogCategory::GAMEPLAY:    return "GAMEPLAY ";
        case LogCategory::AUDIO:       return "AUDIO    ";
        case LogCategory::NETWORKING:  return "NETWORK  ";
        case LogCategory::AI:          return "AI       ";
        case LogCategory::ANIMATION:   return "ANIMATION";
        case LogCategory::PARTICLES:   return "PARTICLES";
        case LogCategory::INPUT:       return "INPUT    ";
        case LogCategory::MEMORY:      return "MEMORY   ";
        case LogCategory::PERFORMANCE: return "PERF     ";
        case LogCategory::ASSET:       return "ASSET    ";
        case LogCategory::SCRIPT:      return "SCRIPT   ";
        case LogCategory::UI:          return "UI       ";
        default:                       return "UNKNOWN  ";
    }
}

std::string Logger::GetColorCode(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE:    return "\033[90m";  // Dark gray
        case LogLevel::DEBUG:    return "\033[36m";  // Cyan
        case LogLevel::INFO:     return "\033[32m";  // Green
        case LogLevel::WARNING:  return "\033[33m";  // Yellow
        case LogLevel::ERROR_LEVEL:    return "\033[31m";  // Red
        case LogLevel::CRITICAL: return "\033[35m";  // Magenta
        case LogLevel::FATAL:    return "\033[91m";  // Bright red
        default:                 return "\033[0m";   // Reset
    }
}

// ScopedTimer implementation
Logger::ScopedTimer::ScopedTimer(const std::string& name, LogCategory category)
    : m_name(name), m_category(category), m_start(std::chrono::high_resolution_clock::now()) {
}

Logger::ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
    
    std::stringstream ss;
    ss << "Timer [" << m_name << "] took " << (duration.count() / 1000.0) << " ms";
    
    Logger::GetInstance().Log(LogLevel::TRACE, m_category, ss.str(), 
                              __FILE__, __LINE__, __FUNCTION__);
}

} // namespace Core
} // namespace CudaGame
