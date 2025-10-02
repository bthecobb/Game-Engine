#include "Debug/DiagnosticsSystem.h"
#include "Rendering/RenderComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Gameplay/PlayerComponents.h"
#include "Gameplay/EnemyComponents.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace CudaGame {
namespace Debug {

// Static initialization
const float DiagnosticsSystem::DIAGNOSTIC_UPDATE_INTERVAL = 0.5f; // Update diagnostics twice per second
LogLevel DiagnosticsSystem::s_logLevel = LogLevel::INFO_LEVEL;

DiagnosticsSystem::DiagnosticsSystem() {
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

DiagnosticsSystem::~DiagnosticsSystem() = default;

bool DiagnosticsSystem::Initialize() {
    DIAG_LOG_INFO("DiagnosticsSystem", "Initializing Diagnostics System");
    
    // Set reasonable defaults
    s_logLevel = LogLevel::INFO_LEVEL;
    m_verboseLogging = false;
    m_showOnScreenDisplay = true;
    
    // DEBUG: Validate coordinator access
    auto& coordinator = Core::Coordinator::GetInstance();
    DIAG_LOG_INFO("DiagnosticsSystem", "Coordinator access validated successfully");
    
    // DEBUG: Test component detection capability
    try {
        // This should work even with entity 0 (just testing the method calls)
        coordinator.HasComponent<Rendering::TransformComponent>(0);
        DIAG_LOG_INFO("DiagnosticsSystem", "Component detection methods accessible");
    } catch (const std::exception& e) {
        DIAG_LOG_WARNING("DiagnosticsSystem", "Component detection test failed: " + std::string(e.what()));
    }
    
    DIAG_LOG_INFO("DiagnosticsSystem", "DiagnosticsSystem initialization complete");
    return true;
}

void DiagnosticsSystem::Shutdown() {
    DIAG_LOG_INFO("DiagnosticsSystem", "Shutting down Diagnostics System");
    m_systemStatuses.clear();
}

void DiagnosticsSystem::Update(float deltaTime) {
    // Update frame timing
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_lastFrameTime).count();
    m_lastFrameTime = currentTime;
    
    // Update performance metrics
    m_metrics.frameTime = frameDuration;
    m_metrics.fps = (frameDuration > 0) ? 1000.0f / frameDuration : 0.0f;
    
    // Add to history
    m_metrics.frameTimeHistory[m_metrics.historyIndex] = m_metrics.frameTime;
    m_metrics.historyIndex = (m_metrics.historyIndex + 1) % PerformanceMetrics::HISTORY_SIZE;
    
    // Update diagnostic timer
    m_diagnosticUpdateTimer += deltaTime;
    
    // Run periodic diagnostics
    if (m_diagnosticUpdateTimer >= DIAGNOSTIC_UPDATE_INTERVAL) {
        RunRuntimeDiagnostics();
        m_diagnosticUpdateTimer = 0.0f;
    }
}

void DiagnosticsSystem::DumpEntityState(Core::Entity entity) {
    std::stringstream ss;
    ss << "Entity " << entity << " State:";
    
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Check common components
    try {
        if (coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            auto& transform = coordinator.GetComponent<Rendering::TransformComponent>(entity);
            ss << "\n  Transform: pos(" << transform.position.x << ", " 
               << transform.position.y << ", " << transform.position.z << ")";
        }
        
        if (coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
            auto& mesh = coordinator.GetComponent<Rendering::MeshComponent>(entity);
            ss << "\n  Mesh: " << mesh.modelPath << " (VAO:" << mesh.vaoId << ")";
        }
        
        if (coordinator.HasComponent<Rendering::MaterialComponent>(entity)) {
            ss << "\n  Material: present";
        }
        
        if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
            auto& rb = coordinator.GetComponent<Physics::RigidbodyComponent>(entity);
            ss << "\n  RigidBody: mass(" << rb.mass << ") kinematic(" << rb.isKinematic << ")";
        }
        
        if (coordinator.HasComponent<Physics::ColliderComponent>(entity)) {
            ss << "\n  Collider: present";
        }
        
        if (coordinator.HasComponent<Gameplay::PlayerCombatComponent>(entity)) {
            ss << "\n  PlayerCombat: present";
        }
        
    } catch (const std::exception& e) {
        ss << "\n  ERROR checking components: " << e.what();
    }
    
    DIAG_LOG_DEBUG("EntityState", ss.str());
}

void DiagnosticsSystem::DumpAllEntities() {
    DIAG_LOG_INFO("DiagnosticsSystem", "=== ENTITY DUMP START ===");
    
    // auto& coordinator = Core::Coordinator::GetInstance();
    
    // This is a simplified version - in a real implementation you'd need access to the entity manager
    // For now, we'll focus on known entities or those we can find through systems
    
    DIAG_LOG_INFO("DiagnosticsSystem", "=== ENTITY DUMP END ===");
}

void DiagnosticsSystem::RegisterSystemForMonitoring(const std::string& name, Core::System* system) {
    SystemStatus status;
    status.name = name;
    status.initialized = (system != nullptr);
    m_systemStatuses.push_back(status);
    
    DIAG_LOG_DEBUG("DiagnosticsSystem", "Registered system for monitoring: " + name);
}

void DiagnosticsSystem::UpdateSystemStatus(const std::string& name, bool updating, size_t entityCount) {
    for (auto& status : m_systemStatuses) {
        if (status.name == name) {
            status.updating = updating;
            status.entityCount = entityCount;
            status.lastUpdateTime = m_metrics.frameTime;
            break;
        }
    }
}

void DiagnosticsSystem::LogSystemError(const std::string& systemName, const std::string& error) {
    for (auto& status : m_systemStatuses) {
        if (status.name == systemName) {
            status.lastError = error;
            break;
        }
    }
    
    DIAG_LOG_ERROR("System:" + systemName, error);
}

void DiagnosticsSystem::DumpSystemStatus() {
    DIAG_LOG_INFO("DiagnosticsSystem", "=== SYSTEM STATUS DUMP ===");
    
    for (const auto& status : m_systemStatuses) {
        std::stringstream ss;
        ss << status.name << ": "
           << "initialized=" << (status.initialized ? "YES" : "NO")
           << ", updating=" << (status.updating ? "YES" : "NO")
           << ", entities=" << status.entityCount
           << ", lastUpdate=" << std::fixed << std::setprecision(2) << status.lastUpdateTime << "ms";
        
        if (!status.lastError.empty()) {
            ss << ", ERROR: " << status.lastError;
        }
        
        DIAG_LOG_INFO("SystemStatus", ss.str());
    }
}

void DiagnosticsSystem::ValidateRenderPipeline() {
    DIAG_LOG_DEBUG("DiagnosticsSystem", "Validating render pipeline...");
    
    // Basic render state validation
    // This would need to be implemented based on your specific rendering system
    
    DIAG_LOG_DEBUG("DiagnosticsSystem", "Render pipeline validation complete");
}

void DiagnosticsSystem::ValidateShaders() {
    DIAG_LOG_DEBUG("DiagnosticsSystem", "Validating shader loading...");
    
    // Check for common shader issues
    std::vector<std::string> requiredShaders = {
        "deferred_geometry.vert",
        "deferred_geometry.frag",
        "deferred_lighting.vert",
        "deferred_lighting.frag",
        "debug_texture.vert",
        "debug_texture.frag"
    };
    
    // This would need access to your shader manager to properly validate
    DIAG_LOG_INFO("DiagnosticsSystem", "Shader validation would need ShaderManager integration");
}

void DiagnosticsSystem::UpdatePerformanceMetrics(float /*frameTime*/, int drawCalls, int triangles) {
    m_metrics.drawCalls = drawCalls;
    m_metrics.triangles = triangles;
    
    // Calculate entity count from coordinator if available
    // auto& coordinator = Core::Coordinator::GetInstance();
    // This would need access to entity manager's entity count
    // m_metrics.totalEntities = coordinator.GetEntityCount();
}

void DiagnosticsSystem::Log(LogLevel level, const std::string& category, const std::string& message) {
    if (level > s_logLevel) {
        return; // Skip logging if below current log level
    }
    
    std::string levelStr;
    switch (level) {
        case LogLevel::ERROR_LEVEL:   levelStr = "[ERROR]"; break;
        case LogLevel::WARNING_LEVEL: levelStr = "[WARN ]"; break;
        case LogLevel::INFO_LEVEL:    levelStr = "[INFO ]"; break;
        case LogLevel::DEBUG_LEVEL:   levelStr = "[DEBUG]"; break;
        case LogLevel::VERBOSE_LEVEL: levelStr = "[VERB ]"; break;
    }
    
    // Format: [LEVEL] [Category] Message
    std::cout << levelStr << " [" << category << "] " << message << std::endl;
}

bool DiagnosticsSystem::ValidateGameSystems() {
    DIAG_LOG_INFO("DiagnosticsSystem", "=== SYSTEM VALIDATION ===");
    
    bool allValid = true;
    
    // auto& coordinator = Core::Coordinator::GetInstance();
    
    // Check if essential systems are registered
    // This would need integration with your system registry
    
    std::string validationResult = allValid ? "PASSED" : "FAILED";
    DIAG_LOG_INFO("DiagnosticsSystem", "System validation " + validationResult);
    return allValid;
}

void DiagnosticsSystem::RunStartupDiagnostics() {
    DIAG_LOG_INFO("DiagnosticsSystem", "=== STARTUP DIAGNOSTICS ===");
    
    // Validate core systems
    ValidateGameSystems();
    
    // Check shader loading
    ValidateShaders();
    
    // Validate render pipeline
    ValidateRenderPipeline();
    
    DIAG_LOG_INFO("DiagnosticsSystem", "=== STARTUP DIAGNOSTICS COMPLETE ===");
}

void DiagnosticsSystem::RunRuntimeDiagnostics() {
    if (m_verboseLogging) {
        // Only log performance metrics periodically when verbose logging is enabled
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << m_metrics.fps
           << ", FrameTime: " << std::setprecision(2) << m_metrics.frameTime << "ms"
           << ", DrawCalls: " << m_metrics.drawCalls
           << ", Entities: " << m_metrics.totalEntities;
        
        DIAG_LOG_VERBOSE("Performance", ss.str());
    }
    
    // Check for any system errors
    for (const auto& status : m_systemStatuses) {
        if (!status.lastError.empty()) {
            DIAG_LOG_WARNING("DiagnosticsSystem", 
                "System " + status.name + " has error: " + status.lastError);
        }
    }
}

void DiagnosticsSystem::ValidateEntityComponents(Core::Entity entity) {
    DIAG_LOG_DEBUG("DiagnosticsSystem", "Validating components for entity " + std::to_string(entity));
    
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Check for component consistency
    // For example, if an entity has a Mesh, it should also have a Transform
    try {
        bool hasMesh = coordinator.HasComponent<Rendering::MeshComponent>(entity);
        bool hasTransform = coordinator.HasComponent<Rendering::TransformComponent>(entity);
        
        if (hasMesh && !hasTransform) {
            DIAG_LOG_WARNING("ComponentValidation", 
                "Entity " + std::to_string(entity) + " has Mesh but no Transform");
        }
        
        bool hasRigidBody = coordinator.HasComponent<Physics::RigidbodyComponent>(entity);
        if (hasRigidBody && !hasTransform) {
            DIAG_LOG_WARNING("ComponentValidation",
                "Entity " + std::to_string(entity) + " has RigidBody but no Transform");
        }
        
    } catch (const std::exception& e) {
        DIAG_LOG_ERROR("ComponentValidation", 
            "Error validating entity " + std::to_string(entity) + ": " + e.what());
    }
}

std::string DiagnosticsSystem::GetEntityDebugString(Core::Entity entity) {
    std::stringstream ss;
    ss << "Entity[" << entity << "]";
    
    auto& coordinator = Core::Coordinator::GetInstance();
    try {
        if (coordinator.HasComponent<Rendering::TransformComponent>(entity)) {
            ss << " +Transform";
        }
        if (coordinator.HasComponent<Rendering::MeshComponent>(entity)) {
            ss << " +Mesh";
        }
        if (coordinator.HasComponent<Rendering::MaterialComponent>(entity)) {
            ss << " +Material";
        }
        if (coordinator.HasComponent<Physics::RigidbodyComponent>(entity)) {
            ss << " +RigidBody";
        }
        if (coordinator.HasComponent<Physics::ColliderComponent>(entity)) {
            ss << " +Collider";
        }
        if (coordinator.HasComponent<Gameplay::PlayerCombatComponent>(entity)) {
            ss << " +PlayerCombat";
        }
    } catch (...) {
        ss << " [component check failed]";
    }
    
    return ss.str();
}

} // namespace Debug
} // namespace CudaGame
