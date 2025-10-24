#pragma once

#include "Combat/CombatMove.h"
#include "Combat/CombatSystem.h"
#include "Input/InputSystem.h"
#include "Audio/AudioSystem.h"
#include "GameFeel/GameFeelSystem.h"
#include "Core/ECS_Types.h"
#include <glm/glm.hpp>
#include <unordered_map>
#include <queue>
#include <memory>

namespace CudaGame {
namespace Combat {

// ComboState is defined in CombatSystem.h

// Input buffer entry for precise timing
struct BufferedInput {
    std::string moveName;
    float timestamp;
    float bufferDuration = 0.2f; // How long this input stays buffered
};

// Combo execution context for an entity
struct ComboContext {
    Core::Entity entityID;
    ComboState currentState = ComboState::Idle;
    
    // Current move execution
    std::shared_ptr<Move> currentMove = nullptr;
    float moveStartTime = 0.0f;
    float currentMoveTime = 0.0f;
    int currentFrame = 0;
    
    // Combo tracking
    std::shared_ptr<Combo> activeCombo = nullptr;
    const Combo::ComboNode* currentComboNode = nullptr;
    int comboCounter = 0;
    float comboTimer = 0.0f;
    float comboResetTime = 2.0f; // Time before combo resets
    
    // Input buffering
    std::queue<BufferedInput> inputBuffer;
    float lastInputTime = 0.0f;
    
    // Hit confirmation
    bool lastMoveHit = false;
    float hitstopTimer = 0.0f;
    
    // Rhythm integration
    bool rhythmWindowActive = false;
    float rhythmAccuracy = 0.0f; // 0.0 to 1.0, how well timed the input was
};

// Manages combo execution and input processing
class ComboManager {
public:
    ComboManager();
    ~ComboManager() = default;
    
    // System lifecycle
    void Initialize(Input::InputSystem* inputSystem, Audio::AudioSystem* audioSystem, GameFeel::GameFeelSystem* gameFeelSystem);
    void Update(float deltaTime);
    void Shutdown();
    
    // Move and combo registration
    void RegisterMove(const std::string& name, std::shared_ptr<Move> move);
    void RegisterCombo(const std::string& name, std::shared_ptr<Combo> combo);
    std::shared_ptr<Move> GetMove(const std::string& name) const;
    std::shared_ptr<Combo> GetCombo(const std::string& name) const;
    
    // Entity combo management
    void CreateComboContext(Core::Entity entityID);
    void DestroyComboContext(Core::Entity entityID);
    ComboContext* GetComboContext(Core::Entity entityID);
    const ComboContext* GetComboContext(Core::Entity entityID) const;
    
    // Combat execution
    bool TryExecuteMove(Core::Entity entityID, const std::string& moveName, float currentTime);
    bool TryExecuteCombo(Core::Entity entityID, const std::string& comboName, float currentTime);
    void CancelCurrentMove(Core::Entity entityID, float currentTime);
    void CompleteCurrentMove(Core::Entity entityID, float currentTime);
    
    // Input processing
    void ProcessInput(Core::Entity entityID, const std::string& inputAction, float currentTime);
    void BufferInput(Core::Entity entityID, const std::string& moveName, float currentTime);
    void ProcessBufferedInputs(Core::Entity entityID, float currentTime);
    
    // Combo state queries
    bool CanCancel(Core::Entity entityID, float currentTime) const;
    bool IsInCombo(Core::Entity entityID) const;
    ComboState GetComboState(Core::Entity entityID) const;
    int GetComboCount(Core::Entity entityID) const;
    
    // Hit confirmation and effects
    void OnHitLanded(Core::Entity attackerID, Core::Entity targetID, float damage, float currentTime);
    void OnHitBlocked(Core::Entity attackerID, Core::Entity targetID, float currentTime);
    void OnHitMissed(Core::Entity attackerID, float currentTime);
    
    // Rhythm integration
    void UpdateRhythmWindows(float currentTime);
    float CalculateRhythmBonus(Core::Entity entityID, float inputTime) const;
    
    // Debug and visualization
    void DebugDrawComboState(Core::Entity entityID) const;
    void GetComboStats(Core::Entity entityID, int& comboCount, float& comboDamage) const;

private:
    // Internal update methods
    void UpdateComboContexts(float deltaTime, float currentTime);
    void UpdateHitstop(ComboContext& context, float deltaTime);
    void UpdateMoveExecution(ComboContext& context, float currentTime);
    void UpdateComboTimer(ComboContext& context, float deltaTime);
    void CheckForAutoCancel(ComboContext& context, float currentTime);
    
    // Move execution helpers
    bool CanExecuteMove(const ComboContext& context, const std::string& moveName) const;
    void StartMove(ComboContext& context, std::shared_ptr<Move> move, float currentTime);
    void AdvanceCombo(ComboContext& context, std::shared_ptr<Move> nextMove);
    void ResetCombo(ComboContext& context);
    
    // Input validation
    bool IsValidComboTransition(const ComboContext& context, const std::string& moveName) const;
    float GetInputBufferWindow() const { return 0.2f; } // 200ms buffer window
    float GetCancelWindow(const ComboContext& context) const;
    
    // Effect helpers
    void PlayMoveEffects(const Move& move, const FrameData& frameData, Core::Entity entityID);
    void ApplyHitstop(ComboContext& context, float duration);
    void TriggerScreenShake(float intensity);
    
private:
    // Move and combo databases
    std::unordered_map<std::string, std::shared_ptr<Move>> m_moves;
    std::unordered_map<std::string, std::shared_ptr<Combo>> m_combos;
    
    // Active combo contexts
    std::unordered_map<Core::Entity, ComboContext> m_comboContexts;
    
    // System references
    Input::InputSystem* m_inputSystem = nullptr;
    Audio::AudioSystem* m_audioSystem = nullptr;
    GameFeel::GameFeelSystem* m_gameFeelSystem = nullptr;
    
    // Global settings
    float m_globalHitstopMultiplier = 1.0f;
    bool m_enableInputBuffer = true;
    bool m_enableRhythmSystem = true;
    
    // Debug settings
    bool m_debugVisualization = false;
    bool m_showFrameData = false;
};

} // namespace Combat
} // namespace CudaGame
