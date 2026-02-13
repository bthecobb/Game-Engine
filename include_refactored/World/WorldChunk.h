#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <atomic>
#include "Core/ECS_Types.h"

namespace CudaGame {
namespace World {

// Forward declarations
struct BuildingMesh;

/**
 * @brief Chunk state for streaming
 */
enum class ChunkState {
    UNLOADED,       // Not in memory
    QUEUED,         // Queued for loading
    LOADING,        // Being generated on worker thread
    LOADED,         // Ready for rendering
    UNLOADING       // Being freed
};

/**
 * @brief Axis-aligned bounding box for culling
 */
struct AABB {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    
    bool Contains(const glm::vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
    
    bool Intersects(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y) &&
               (min.z <= other.max.z && max.z >= other.min.z);
    }
    
    glm::vec3 GetCenter() const {
        return (min + max) * 0.5f;
    }
    
    glm::vec3 GetExtents() const {
        return (max - min) * 0.5f;
    }
};

/**
 * @brief LOD level for chunks
 */
enum class ChunkLOD {
    HIGH = 0,       // Full detail (close to player)
    MEDIUM = 1,     // Reduced detail
    LOW = 2         // Minimal detail (far from player)
};

/**
 * @brief Configuration for chunk system
 */
struct ChunkConfig {
    static constexpr float CHUNK_SIZE = 2048.0f;        // Units per chunk
    static constexpr int ACTIVE_RADIUS = 3;             // Chunks in each direction
    static constexpr int LOD_LEVELS = 2;                // 0, 1, 2
    static constexpr int BUILDINGS_PER_CHUNK = 40;      // Average buildings
    static constexpr float LOD_DISTANCE_0 = 2048.0f;    // High detail distance
    static constexpr float LOD_DISTANCE_1 = 4096.0f;    // Medium detail distance
    // Beyond LOD_DISTANCE_1 = Low detail
};

/**
 * @brief Represents a single world chunk
 * 
 * Contains all entities and meshes for a grid cell
 */
struct WorldChunk {
    // Grid coordinates
    glm::ivec2 coord{0, 0};
    
    // World-space bounds
    AABB bounds;
    
    // Current state (thread safety handled by manager's mutex)
    ChunkState state{ChunkState::UNLOADED};
    
    // Current LOD level
    ChunkLOD lodLevel{ChunkLOD::HIGH};
    
    // Deterministic seed for procedural generation
    uint32_t seed{0};
    
    // Generated entities (buildings, props, etc.)
    std::vector<Core::Entity> entities;
    
    // Entity mesh indices (for GPU buffer management)
    std::vector<size_t> meshIndices;
    
    // Building count at each LOD
    int buildingCounts[3] = {0, 0, 0};
    
    // Distance from player (updated each frame)
    float distanceToPlayer{0.0f};
    
    // Frame last accessed (for LRU unloading)
    uint64_t lastAccessFrame{0};
    
    /**
     * @brief Calculate chunk's world-space center
     */
    glm::vec3 GetWorldCenter() const {
        return glm::vec3(
            coord.x * ChunkConfig::CHUNK_SIZE + ChunkConfig::CHUNK_SIZE * 0.5f,
            0.0f,
            coord.y * ChunkConfig::CHUNK_SIZE + ChunkConfig::CHUNK_SIZE * 0.5f
        );
    }
    
    /**
     * @brief Get deterministic seed for this chunk
     */
    static uint32_t CalculateSeed(int x, int y) {
        // Use a hash combining function for deterministic seeding
        uint32_t seed = 12345;
        seed ^= static_cast<uint32_t>(x) * 73856093;
        seed ^= static_cast<uint32_t>(y) * 19349663;
        return seed;
    }
    
    /**
     * @brief Reset chunk for reuse
     */
    void Reset() {
        state = ChunkState::UNLOADED;
        lodLevel = ChunkLOD::HIGH;
        entities.clear();
        meshIndices.clear();
        buildingCounts[0] = buildingCounts[1] = buildingCounts[2] = 0;
        distanceToPlayer = 0.0f;
        lastAccessFrame = 0;
    }
};

} // namespace World
} // namespace CudaGame
