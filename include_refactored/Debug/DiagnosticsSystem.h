#pragma once

#include "Core/System.h"
#include "Core/Coordinator.h"
#include "Core/ECS_Types.h"
#include <string>
#include <vector>
#include <chrono>

namespace CudaGame {
namespace Debug {

enum class LogLevel : int {
    ERROR_LEVEL = 0,
    WARNING_LEVEL = 1,
    INFO_LEVEL = 2,
    DEBUG_LEVEL = 3,
    VERBOSE_LEVEL = 4
};

struct SystemStatus {
    std::string name;
    bool initialized = false;
    bool updating = false;
    size_t entityCount = 0;
    float lastUpdateTime = 0.0f;
    std::string lastError;
};

struct PerformanceMetrics {
    float fps = 0.0f;
    float frameTime = 0.0f;
    int drawCalls = 0;
    int triangles = 0;
    size_t totalEntities = 0;
    size_t memoryUsage = 0;
    
    // Frame time history for graphing
    static const int HISTORY_SIZE = 120;
    float frameTimeHistory[HISTORY_SIZE] = {0.0f};
    int historyIndex = 0;
};

class DiagnosticsSystem : public Core::System {
public:
    DiagnosticsSystem();
    ~DiagnosticsSystem();

    bool Initialize() override;
    void Shutdown() override;
    void Update(float deltaTime) override;

    // Entity diagnostics
    void DumpEntityState(Core::Entity entity);
    void DumpAllEntities();
    void ValidateEntityComponents(Core::Entity entity);
    
    // System diagnostics
    void RegisterSystemForMonitoring(const std::string& name, Core::System* system);
    void UpdateSystemStatus(const std::string& name, bool updating, size_t entityCount);
    void LogSystemError(const std::string& systemName, const std::string& error);
    void ValidateSystemConnections();
    void DumpSystemStatus();
    
    // Rendering diagnostics
    void ValidateRenderPipeline();
    void DumpRenderState();
    void ValidateShaders();
    void CheckMeshLoading();
    
    // Performance monitoring
    void UpdatePerformanceMetrics(float frameTime, int drawCalls, int triangles);
    const PerformanceMetrics& GetPerformanceMetrics() const { return m_metrics; }
    
    // Logging system
    static void SetLogLevel(LogLevel level) { s_logLevel = level; }
    static LogLevel GetLogLevel() { return s_logLevel; }
    static void Log(LogLevel level, const std::string& category, const std::string& message);
    
    // Debug controls
    void ToggleVerboseLogging() { m_verboseLogging = !m_verboseLogging; }
    void ToggleOnScreenDisplay() { m_showOnScreenDisplay = !m_showOnScreenDisplay; }
    bool IsOnScreenDisplayEnabled() const { return m_showOnScreenDisplay; }
    
    // Validation functions
    bool ValidateGameSystems();
    void RunStartupDiagnostics();
    void RunRuntimeDiagnostics();

private:
    // Internal tracking
    std::vector<SystemStatus> m_systemStatuses;
    PerformanceMetrics m_metrics;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    float m_diagnosticUpdateTimer = 0.0f;
    static const float DIAGNOSTIC_UPDATE_INTERVAL;
    
    // Settings
    bool m_verboseLogging = false;
    bool m_showOnScreenDisplay = true;
    static LogLevel s_logLevel;
    
    // Helper methods
    void LogComponentState(Core::Entity entity, const std::string& componentName);
    std::string GetEntityDebugString(Core::Entity entity);
    void CheckForOrphanedEntities();
    void ValidateComponentIntegrity();
};

// Convenience macros for diagnostics logging
#define DIAG_LOG_ERROR(category, message) \
    CudaGame::Debug::DiagnosticsSystem::Log(CudaGame::Debug::LogLevel::ERROR_LEVEL, category, message)

#define DIAG_LOG_WARNING(category, message) \
    CudaGame::Debug::DiagnosticsSystem::Log(CudaGame::Debug::LogLevel::WARNING_LEVEL, category, message)

#define DIAG_LOG_INFO(category, message) \
    CudaGame::Debug::DiagnosticsSystem::Log(CudaGame::Debug::LogLevel::INFO_LEVEL, category, message)

#define DIAG_LOG_DEBUG(category, message) \
    CudaGame::Debug::DiagnosticsSystem::Log(CudaGame::Debug::LogLevel::DEBUG_LEVEL, category, message)

#define DIAG_LOG_VERBOSE(category, message) \
    CudaGame::Debug::DiagnosticsSystem::Log(CudaGame::Debug::LogLevel::VERBOSE_LEVEL, category, message)

} // namespace Debug
} // namespace CudaGame
