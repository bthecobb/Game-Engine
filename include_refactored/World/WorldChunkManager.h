#pragma once

#include "World/WorldChunk.h"
#include "Core/ThreadPool.h"
#include <unordered_map>
#include <vector>
#include <queue>
#include <mutex>
#include <functional>
#include <glm/glm.hpp>

namespace CudaGame {
namespace World {

/**
 * @brief Hash function for chunk coordinates
 */
struct ChunkCoordHash {
    size_t operator()(const glm::ivec2& coord) const {
        return std::hash<int>()(coord.x) ^ (std::hash<int>()(coord.y) << 16);
    }
};

/**
 * @brief Manages chunk loading, unloading, and streaming
 * 
 * Uses distance-based streaming similar to UE5 World Partition:
 * - Chunks within active radius are loaded
 * - Chunks outside are queued for unload
 * - LOD is determined by distance to player
 */
class WorldChunkManager {
public:
    // Callback types for chunk events
    using ChunkGenerateFunc = std::function<void(WorldChunk&)>;
    using ChunkLoadedFunc = std::function<void(WorldChunk&)>;
    using ChunkUnloadedFunc = std::function<void(WorldChunk&)>;
    
    WorldChunkManager();
    ~WorldChunkManager();
    
    /**
     * @brief Initialize the chunk manager
     * @param generateCallback Called on worker thread to generate chunk content
     * @param loadedCallback Called on main thread when chunk is ready
     * @param unloadedCallback Called on main thread before chunk is freed
     */
    void Initialize(
        ChunkGenerateFunc generateCallback,
        ChunkLoadedFunc loadedCallback,
        ChunkUnloadedFunc unloadedCallback
    );
    
    /**
     * @brief Shutdown and cleanup
     */
    void Shutdown();
    
    /**
     * @brief Update chunk streaming based on player position
     * @param playerPosition Current player world position
     * @param frameNumber Current frame for LRU tracking
     */
    void Update(const glm::vec3& playerPosition, uint64_t frameNumber);
    
    /**
     * @brief Process completed chunk loads on main thread
     * Must be called from main thread each frame
     */
    void ProcessCompletedChunks();
    
    /**
     * @brief Get chunk at world position (may be null if not loaded)
     */
    WorldChunk* GetChunkAtPosition(const glm::vec3& worldPos);
    
    /**
     * @brief Get chunk at grid coordinates
     */
    WorldChunk* GetChunk(const glm::ivec2& coord);
    
    /**
     * @brief Get all currently loaded chunks
     */
    const std::unordered_map<glm::ivec2, std::unique_ptr<WorldChunk>, ChunkCoordHash>& 
        GetLoadedChunks() const { return m_chunks; }
    
    /**
     * @brief Get current player chunk coordinates
     */
    glm::ivec2 GetPlayerChunkCoord() const { return m_playerChunkCoord; }
    
    /**
     * @brief Convert world position to chunk coordinates
     */
    static glm::ivec2 WorldToChunkCoord(const glm::vec3& worldPos);
    
    /**
     * @brief Get world bounds for a chunk
     */
    static AABB GetChunkBounds(const glm::ivec2& coord);
    
    /**
     * @brief Get total number of loaded chunks
     */
    size_t GetLoadedChunkCount() const { return m_chunks.size(); }
    
    /**
     * @brief Get number of chunks currently loading
     */
    size_t GetLoadingChunkCount() const { return m_loadingCount; }
    
    /**
     * @brief Force load specific chunks (for debugging/testing)
     */
    void ForceLoadChunk(const glm::ivec2& coord);
    
    /**
     * @brief Force unload specific chunk
     */
    void ForceUnloadChunk(const glm::ivec2& coord);

private:
    // Chunk storage
    std::unordered_map<glm::ivec2, std::unique_ptr<WorldChunk>, ChunkCoordHash> m_chunks;
    
    // Chunks completed on worker threads, ready for main thread processing
    std::queue<std::shared_ptr<WorldChunk>> m_completedChunks;
    std::mutex m_completedMutex;
    
    // Chunks queued for unloading
    std::queue<glm::ivec2> m_unloadQueue;
    std::mutex m_unloadMutex;
    
    // Current player chunk position
    glm::ivec2 m_playerChunkCoord{0, 0};
    glm::vec3 m_playerPosition{0.0f};
    
    // Tracking
    std::atomic<size_t> m_loadingCount{0};
    uint64_t m_currentFrame{0};
    
    // Callbacks
    ChunkGenerateFunc m_generateCallback;
    ChunkLoadedFunc m_loadedCallback;
    ChunkUnloadedFunc m_unloadedCallback;
    
    // State
    bool m_initialized{false};
    
    // Internal methods
    void UpdateChunkLoading();
    void UpdateChunkUnloading();
    void UpdateChunkLOD();
    ChunkLOD CalculateLOD(float distance) const;
    bool IsChunkInActiveRange(const glm::ivec2& coord) const;
    void QueueChunkLoad(const glm::ivec2& coord);
    void QueueChunkUnload(const glm::ivec2& coord);
    void GenerateChunkAsync(std::unique_ptr<WorldChunk> chunk);
};

} // namespace World
} // namespace CudaGame
