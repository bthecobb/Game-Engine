#include "World/WorldChunkManager.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace CudaGame {
namespace World {

WorldChunkManager::WorldChunkManager() {}

WorldChunkManager::~WorldChunkManager() {
    Shutdown();
}

void WorldChunkManager::Initialize(
    ChunkGenerateFunc generateCallback,
    ChunkLoadedFunc loadedCallback,
    ChunkUnloadedFunc unloadedCallback)
{
    m_generateCallback = generateCallback;
    m_loadedCallback = loadedCallback;
    m_unloadedCallback = unloadedCallback;
    m_initialized = true;
    
    std::cout << "[ChunkManager] Initialized with chunk size " 
              << ChunkConfig::CHUNK_SIZE << ", active radius " 
              << ChunkConfig::ACTIVE_RADIUS << std::endl;
}

void WorldChunkManager::Shutdown() {
    if (!m_initialized) return;
    
    std::cout << "[ChunkManager] Shutting down, unloading " 
              << m_chunks.size() << " chunks..." << std::endl;
    
    // Unload all chunks
    for (auto& pair : m_chunks) {
        if (m_unloadedCallback && pair.second) {
            m_unloadedCallback(*pair.second);
        }
    }
    m_chunks.clear();
    
    // Clear queues
    {
        std::lock_guard<std::mutex> lock(m_completedMutex);
        while (!m_completedChunks.empty()) m_completedChunks.pop();
    }
    
    m_initialized = false;
}

void WorldChunkManager::Update(const glm::vec3& playerPosition, uint64_t frameNumber) {
    if (!m_initialized) return;
    
    m_playerPosition = playerPosition;
    m_currentFrame = frameNumber;
    
    // Calculate current chunk
    glm::ivec2 newChunkCoord = WorldToChunkCoord(playerPosition);
    bool playerChangedChunk = (newChunkCoord != m_playerChunkCoord);
    m_playerChunkCoord = newChunkCoord;
    
    // Update distances for all loaded chunks
    for (auto& pair : m_chunks) {
        WorldChunk& chunk = *pair.second;
        glm::vec3 chunkCenter = chunk.GetWorldCenter();
        chunk.distanceToPlayer = glm::distance(
            glm::vec2(playerPosition.x, playerPosition.z),
            glm::vec2(chunkCenter.x, chunkCenter.z)
        );
        chunk.lastAccessFrame = frameNumber;
    }
    
    // If player changed chunks or streaming needed, update loading/unloading
    UpdateChunkLoading();
    UpdateChunkUnloading();
    UpdateChunkLOD();
}

void WorldChunkManager::ProcessCompletedChunks() {
    std::lock_guard<std::mutex> lock(m_completedMutex);
    
    while (!m_completedChunks.empty()) {
        std::shared_ptr<WorldChunk> chunk = std::move(m_completedChunks.front());
        m_completedChunks.pop();
        
        if (chunk && chunk->state == ChunkState::LOADING) {
            chunk->state = ChunkState::LOADED;
            
            // Call loaded callback on main thread
            if (m_loadedCallback) {
                m_loadedCallback(*chunk);
            }
            
            // Store in map - convert shared_ptr to unique_ptr
            // Since we're the only owner at this point, we can safely take ownership
            glm::ivec2 coord = chunk->coord;
            m_chunks[coord] = std::make_unique<WorldChunk>();
            // Copy the data (WorldChunk needs to be movable for this)
            *m_chunks[coord] = std::move(*chunk);
            --m_loadingCount;
            
            std::cout << "[ChunkManager] Chunk (" << coord.x << ", " << coord.y 
                      << ") loaded" << std::endl;
        }
    }
}

WorldChunk* WorldChunkManager::GetChunkAtPosition(const glm::vec3& worldPos) {
    glm::ivec2 coord = WorldToChunkCoord(worldPos);
    return GetChunk(coord);
}

WorldChunk* WorldChunkManager::GetChunk(const glm::ivec2& coord) {
    auto it = m_chunks.find(coord);
    return (it != m_chunks.end()) ? it->second.get() : nullptr;
}

glm::ivec2 WorldChunkManager::WorldToChunkCoord(const glm::vec3& worldPos) {
    return glm::ivec2(
        static_cast<int>(std::floor(worldPos.x / ChunkConfig::CHUNK_SIZE)),
        static_cast<int>(std::floor(worldPos.z / ChunkConfig::CHUNK_SIZE))
    );
}

AABB WorldChunkManager::GetChunkBounds(const glm::ivec2& coord) {
    AABB bounds;
    bounds.min = glm::vec3(
        coord.x * ChunkConfig::CHUNK_SIZE,
        -100.0f,  // Below ground
        coord.y * ChunkConfig::CHUNK_SIZE
    );
    bounds.max = glm::vec3(
        (coord.x + 1) * ChunkConfig::CHUNK_SIZE,
        500.0f,   // Max building height
        (coord.y + 1) * ChunkConfig::CHUNK_SIZE
    );
    return bounds;
}

void WorldChunkManager::ForceLoadChunk(const glm::ivec2& coord) {
    if (m_chunks.find(coord) == m_chunks.end()) {
        QueueChunkLoad(coord);
    }
}

void WorldChunkManager::ForceUnloadChunk(const glm::ivec2& coord) {
    QueueChunkUnload(coord);
}

bool WorldChunkManager::IsChunkInActiveRange(const glm::ivec2& coord) const {
    int dx = std::abs(coord.x - m_playerChunkCoord.x);
    int dy = std::abs(coord.y - m_playerChunkCoord.y);
    return dx <= ChunkConfig::ACTIVE_RADIUS && dy <= ChunkConfig::ACTIVE_RADIUS;
}

ChunkLOD WorldChunkManager::CalculateLOD(float distance) const {
    if (distance < ChunkConfig::LOD_DISTANCE_0) {
        return ChunkLOD::HIGH;
    } else if (distance < ChunkConfig::LOD_DISTANCE_1) {
        return ChunkLOD::MEDIUM;
    }
    return ChunkLOD::LOW;
}

void WorldChunkManager::UpdateChunkLoading() {
    // Check all chunks in active range
    for (int dx = -ChunkConfig::ACTIVE_RADIUS; dx <= ChunkConfig::ACTIVE_RADIUS; ++dx) {
        for (int dy = -ChunkConfig::ACTIVE_RADIUS; dy <= ChunkConfig::ACTIVE_RADIUS; ++dy) {
            glm::ivec2 coord = m_playerChunkCoord + glm::ivec2(dx, dy);
            
            // Skip if already loaded or loading
            auto it = m_chunks.find(coord);
            if (it != m_chunks.end()) {
                continue;
            }
            
            // Queue for loading
            QueueChunkLoad(coord);
        }
    }
}

void WorldChunkManager::UpdateChunkUnloading() {
    // Find chunks outside active range
    std::vector<glm::ivec2> toUnload;
    
    for (auto& pair : m_chunks) {
        if (!IsChunkInActiveRange(pair.first)) {
            toUnload.push_back(pair.first);
        }
    }
    
    // Unload them
    for (const auto& coord : toUnload) {
        QueueChunkUnload(coord);
    }
}

void WorldChunkManager::UpdateChunkLOD() {
    for (auto& pair : m_chunks) {
        WorldChunk& chunk = *pair.second;
        ChunkLOD newLOD = CalculateLOD(chunk.distanceToPlayer);
        
        if (newLOD != chunk.lodLevel) {
            chunk.lodLevel = newLOD;
            // TODO: Trigger LOD mesh swap
        }
    }
}

void WorldChunkManager::QueueChunkLoad(const glm::ivec2& coord) {
    // Create new chunk
    auto chunk = std::make_unique<WorldChunk>();
    chunk->coord = coord;
    chunk->bounds = GetChunkBounds(coord);
    chunk->seed = WorldChunk::CalculateSeed(coord.x, coord.y);
    chunk->state = ChunkState::QUEUED;
    
    ++m_loadingCount;
    
    // Submit to thread pool
    GenerateChunkAsync(std::move(chunk));
}

void WorldChunkManager::QueueChunkUnload(const glm::ivec2& coord) {
    auto it = m_chunks.find(coord);
    if (it != m_chunks.end()) {
        if (m_unloadedCallback) {
            m_unloadedCallback(*it->second);
        }
        m_chunks.erase(it);
        
        std::cout << "[ChunkManager] Chunk (" << coord.x << ", " << coord.y 
                  << ") unloaded" << std::endl;
    }
}

void WorldChunkManager::GenerateChunkAsync(std::unique_ptr<WorldChunk> chunk) {
    chunk->state = ChunkState::LOADING;
    
    // Move chunk into shared_ptr for thread-safe capture and queue transfer
    auto sharedChunk = std::shared_ptr<WorldChunk>(chunk.release());
    
    Core::ThreadPool::GetInstance().Enqueue([this, sharedChunk]() {
        // Generate chunk content on worker thread
        if (m_generateCallback) {
            m_generateCallback(*sharedChunk);
        }
        
        // Move to completed queue for main thread processing
        {
            std::lock_guard<std::mutex> lock(m_completedMutex);
            m_completedChunks.push(sharedChunk);
        }
    });
}

} // namespace World
} // namespace CudaGame
