#include "Combat/ComboManager.h"
#include <algorithm>
#include <iostream>

namespace CudaGame {
namespace Combat {

ComboManager::ComboManager() {
    // Constructor implementation
}

void ComboManager::Initialize(Input::InputSystem* inputSystem, Audio::AudioSystem* audioSystem, GameFeel::GameFeelSystem* gameFeelSystem) {
    m_inputSystem = inputSystem;
    m_audioSystem = audioSystem;
    m_gameFeelSystem = gameFeelSystem;
}

void ComboManager::Update(float deltaTime) {
    float currentTime = 0.0f; // TODO: Get actual current time from game clock
    UpdateComboContexts(deltaTime, currentTime);
    UpdateRhythmWindows(currentTime);
}

void ComboManager::Shutdown() {
    m_comboContexts.clear();
    m_moves.clear();
    m_combos.clear();
}

// Move and combo registration
void ComboManager::RegisterMove(const std::string& name, std::shared_ptr<Move> move) {
    m_moves[name] = move;
}

void ComboManager::RegisterCombo(const std::string& name, std::shared_ptr<Combo> combo) {
    m_combos[name] = combo;
}

std::shared_ptr<Move> ComboManager::GetMove(const std::string& name) const {
    auto it = m_moves.find(name);
    return (it != m_moves.end()) ? it->second : nullptr;
}

std::shared_ptr<Combo> ComboManager::GetCombo(const std::string& name) const {
    auto it = m_combos.find(name);
    return (it != m_combos.end()) ? it->second : nullptr;
}

// Entity combo management
void ComboManager::CreateComboContext(Core::Entity entityID) {
    ComboContext context;
    context.entityID = entityID;
    m_comboContexts[entityID] = context;
}

void ComboManager::DestroyComboContext(Core::Entity entityID) {
    m_comboContexts.erase(entityID);
}

ComboContext* ComboManager::GetComboContext(Core::Entity entityID) {
    auto it = m_comboContexts.find(entityID);
    return (it != m_comboContexts.end()) ? &it->second : nullptr;
}

// Combat execution
bool ComboManager::TryExecuteMove(Core::Entity entityID, const std::string& moveName, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context || !CanExecuteMove(*context, moveName)) {
        return false;
    }
    
    auto move = GetMove(moveName);
    if (!move) {
        return false;
    }
    
    StartMove(*context, move, currentTime);
    return true;
}

bool ComboManager::TryExecuteCombo(Core::Entity entityID, const std::string& comboName, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    auto combo = GetCombo(comboName);
    
    if (!context || !combo) {
        return false;
    }
    
    // Start the combo with its initial move
    const auto* startingNode = combo->GetStartingNode();
    if (!startingNode) {
        return false;
    }
    
    auto startingMove = GetMove(startingNode->moveName);
    if (!startingMove || !CanExecuteMove(*context, startingNode->moveName)) {
        return false;
    }
    
    // Set up combo context
    context->activeCombo = combo;
    context->currentComboNode = startingNode;
    context->comboCounter = 1;
    context->comboTimer = 0.0f;
    
    StartMove(*context, startingMove, currentTime);
    return true;
}

void ComboManager::CancelCurrentMove(Core::Entity entityID, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context || context->currentState == ComboState::Idle) {
        return;
    }
    
    context->currentState = ComboState::Cancelled;
    context->currentMove = nullptr;
    context->hitstopTimer = 0.0f;
}

void ComboManager::CompleteCurrentMove(Core::Entity entityID, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context) {
        return;
    }
    
    context->currentState = ComboState::Idle;
    context->currentMove = nullptr;
    context->hitstopTimer = 0.0f;
    
    // Reset combo if no follow-up within time limit
    if (context->activeCombo && context->comboTimer > context->comboResetTime) {
        ResetCombo(*context);
    }
}

// Input processing
void ComboManager::ProcessInput(Core::Entity entityID, const std::string& inputAction, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context) {
        return;
    }
    
    // Try to execute the move immediately
    if (TryExecuteMove(entityID, inputAction, currentTime)) {
        context->lastInputTime = currentTime;
        return;
    }
    
    // If immediate execution failed, buffer the input
    if (m_enableInputBuffer) {
        BufferInput(entityID, inputAction, currentTime);
    }
}

void ComboManager::BufferInput(Core::Entity entityID, const std::string& moveName, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context) {
        return;
    }
    
    BufferedInput input;
    input.moveName = moveName;
    input.timestamp = currentTime;
    input.bufferDuration = GetInputBufferWindow();
    
    context->inputBuffer.push(input);
    context->lastInputTime = currentTime;
    
    // Limit buffer size to prevent overflow
    while (context->inputBuffer.size() > 5) {
        context->inputBuffer.pop();
    }
}

void ComboManager::ProcessBufferedInputs(Core::Entity entityID, float currentTime) {
    ComboContext* context = GetComboContext(entityID);
    if (!context) {
        return;
    }
    
    // Process buffered inputs that are still valid
    while (!context->inputBuffer.empty()) {
        const BufferedInput& input = context->inputBuffer.front();
        
        // Check if input has expired
        if (currentTime - input.timestamp > input.bufferDuration) {
            context->inputBuffer.pop();
            continue;
        }
        
        // Try to execute the buffered input
        if (TryExecuteMove(entityID, input.moveName, currentTime)) {
            context->inputBuffer.pop();
            break; // Successfully executed, stop processing
        } else {
            break; // Can't execute yet, keep in buffer
        }
    }
}

// Combo state queries
bool ComboManager::CanCancel(Core::Entity entityID, float currentTime) const {
    const ComboContext* context = GetComboContext(entityID);
    if (!context || !context->currentMove) {
        return false;
    }
    
    try {
        const FrameData& frameData = context->currentMove->GetFrameDataAtTime(context->currentMoveTime);
        return frameData.canCancel;
    } catch (const std::exception&) {
        return false;
    }
}

bool ComboManager::IsInCombo(Core::Entity entityID) const {
    const ComboContext* context = GetComboContext(entityID);
    return context && context->activeCombo != nullptr && context->comboCounter > 0;
}

ComboState ComboManager::GetComboState(Core::Entity entityID) const {
    const ComboContext* context = GetComboContext(entityID);
    return context ? context->currentState : ComboState::Idle;
}

int ComboManager::GetComboCount(Core::Entity entityID) const {
    const ComboContext* context = GetComboContext(entityID);
    return context ? context->comboCounter : 0;
}

// Hit confirmation and effects
void ComboManager::OnHitLanded(Core::Entity attackerID, Core::Entity targetID, float damage, float currentTime) {
    ComboContext* context = GetComboContext(attackerID);
    if (!context || !context->currentMove) {
        return;
    }
    
    context->lastMoveHit = true;
    
    // Apply hitstop effect
    try {
        const FrameData& frameData = context->currentMove->GetFrameDataAtTime(context->currentMoveTime);
        if (frameData.hitstop > 0.0f) {
            ApplyHitstop(*context, frameData.hitstop * m_globalHitstopMultiplier);
        }
        
        // Play hit effects
        PlayMoveEffects(*context->currentMove, frameData, attackerID);
        
        // Trigger screen shake based on damage
        TriggerScreenShake(damage * 0.1f);
        
    } catch (const std::exception&) {
        // Frame data not available, use defaults
        ApplyHitstop(*context, 0.1f);
    }
}

void ComboManager::OnHitBlocked(Core::Entity attackerID, Core::Entity targetID, float currentTime) {
    ComboContext* context = GetComboContext(attackerID);
    if (!context) {
        return;
    }
    
    context->lastMoveHit = false;
    // Apply reduced hitstop for blocked attacks
    ApplyHitstop(*context, 0.05f);
}

void ComboManager::OnHitMissed(Core::Entity attackerID, float currentTime) {
    ComboContext* context = GetComboContext(attackerID);
    if (!context) {
        return;
    }
    
    context->lastMoveHit = false;
    // No hitstop for missed attacks
}

// Rhythm integration
void ComboManager::UpdateRhythmWindows(float currentTime) {
    if (!m_enableRhythmSystem) {
        return;
    }
    
    for (auto& [entityID, context] : m_comboContexts) {
        if (!context.currentMove) {
            context.rhythmWindowActive = false;
            continue;
        }
        
        float windowStart = context.currentMove->GetRhythmWindowStart();
        float windowEnd = context.currentMove->GetRhythmWindowEnd();
        float moveProgress = context.currentMoveTime / context.currentMove->GetTotalFrames();
        
        context.rhythmWindowActive = (moveProgress >= windowStart && moveProgress <= windowEnd);
    }
}

float ComboManager::CalculateRhythmBonus(Core::Entity entityID, float inputTime) const {
    const ComboContext* context = GetComboContext(entityID);
    if (!context || !context->rhythmWindowActive || !m_enableRhythmSystem) {
        return 1.0f; // No bonus
    }
    
    // Calculate timing accuracy (simplified)
    float accuracy = std::max(0.0f, 1.0f - std::abs(context->rhythmAccuracy - 0.5f) * 2.0f);
    return 1.0f + (accuracy * 0.5f); // Up to 50% bonus for perfect timing
}

// Debug and visualization
void ComboManager::DebugDrawComboState(Core::Entity entityID) const {
    if (!m_debugVisualization) {
        return;
    }
    
    const ComboContext* context = GetComboContext(entityID);
    if (!context) {
        return;
    }
    
    std::cout << "Entity " << entityID << " - State: ";
    switch (context->currentState) {
        case ComboState::Idle: std::cout << "Idle"; break;
        case ComboState::Startup: std::cout << "Startup"; break;
        case ComboState::Active: std::cout << "Active"; break;
        case ComboState::Recovery: std::cout << "Recovery"; break;
        case ComboState::Cancelled: std::cout << "Cancelled"; break;
    }
    
    std::cout << ", Combo: " << context->comboCounter;
    if (context->currentMove) {
        std::cout << ", Move: " << context->currentMove->GetName();
        std::cout << ", Frame: " << context->currentFrame;
    }
    std::cout << std::endl;
}

void ComboManager::GetComboStats(Core::Entity entityID, int& comboCount, float& comboDamage) const {
    const ComboContext* context = GetComboContext(entityID);
    if (!context) {
        comboCount = 0;
        comboDamage = 0.0f;
        return;
    }
    
    comboCount = context->comboCounter;
    comboDamage = 0.0f; // TODO: Track accumulated damage
}

// Private helper methods
void ComboManager::UpdateComboContexts(float deltaTime, float currentTime) {
    for (auto& [entityID, context] : m_comboContexts) {
        UpdateHitstop(context, deltaTime);
        UpdateMoveExecution(context, currentTime);
        UpdateComboTimer(context, deltaTime);
        CheckForAutoCancel(context, currentTime);
        ProcessBufferedInputs(entityID, currentTime);
    }
}

void ComboManager::UpdateHitstop(ComboContext& context, float deltaTime) {
    if (context.hitstopTimer > 0.0f) {
        context.hitstopTimer -= deltaTime;
        return; // Don't update other timers during hitstop
    }
    
    context.currentMoveTime += deltaTime;
}

void ComboManager::UpdateMoveExecution(ComboContext& context, float currentTime) {
    if (!context.currentMove || context.hitstopTimer > 0.0f) {
        return;
    }
    
    // Update current frame
    context.currentFrame = static_cast<int>(context.currentMoveTime * 60.0f);
    
    // Update combo state based on frame data
    try {
        const FrameData& frameData = context.currentMove->GetFrameDataAtTime(context.currentMoveTime);
        
        if (frameData.isStartup) {
            context.currentState = ComboState::Startup;
        } else if (frameData.isActive) {
            context.currentState = ComboState::Active;
        } else if (frameData.isRecovery) {
            context.currentState = ComboState::Recovery;
        }
    } catch (const std::exception&) {
        // Frame data not available, move is complete
        CompleteCurrentMove(context.entityID, currentTime);
    }
    
    // Check if move is complete
    if (context.currentFrame >= context.currentMove->GetTotalFrames()) {
        CompleteCurrentMove(context.entityID, currentTime);
    }
}

void ComboManager::UpdateComboTimer(ComboContext& context, float deltaTime) {
    if (context.activeCombo) {
        context.comboTimer += deltaTime;
        
        // Reset combo if timer expires
        if (context.comboTimer > context.comboResetTime) {
            ResetCombo(context);
        }
    }
}

void ComboManager::CheckForAutoCancel(ComboContext& context, float currentTime) {
    // TODO: Implement auto-cancel logic for specific moves
}

bool ComboManager::CanExecuteMove(const ComboContext& context, const std::string& moveName) const {
    // Can always execute if idle
    if (context.currentState == ComboState::Idle) {
        return true;
    }
    
    // Check if we can cancel current move
    if (!CanCancel(context.entityID, 0.0f)) {
        return false;
    }
    
    // Check if this is a valid combo transition
    return IsValidComboTransition(context, moveName);
}

void ComboManager::StartMove(ComboContext& context, std::shared_ptr<Move> move, float currentTime) {
    context.currentMove = move;
    context.moveStartTime = currentTime;
    context.currentMoveTime = 0.0f;
    context.currentFrame = 0;
    context.currentState = ComboState::Startup;
    context.lastMoveHit = false;
    
    // Advance combo if we're in one
    if (context.activeCombo) {
        AdvanceCombo(context, move);
    }
}

void ComboManager::AdvanceCombo(ComboContext& context, std::shared_ptr<Move> nextMove) {
    if (!context.activeCombo || !context.currentComboNode) {
        return;
    }
    
    // Find the next combo node
    const auto* nextNode = context.activeCombo->GetNode(nextMove->GetName());
    if (nextNode) {
        context.currentComboNode = nextNode;
        context.comboCounter++;
        context.comboTimer = 0.0f; // Reset combo timer
    }
}

void ComboManager::ResetCombo(ComboContext& context) {
    context.activeCombo = nullptr;
    context.currentComboNode = nullptr;
    context.comboCounter = 0;
    context.comboTimer = 0.0f;
}

bool ComboManager::IsValidComboTransition(const ComboContext& context, const std::string& moveName) const {
    if (!context.activeCombo || !context.currentComboNode) {
        return true; // Not in a combo, any move is valid
    }
    
    // Check if the move is in the list of valid next moves
    const auto& nextMoves = context.currentComboNode->nextMoves;
    return std::find(nextMoves.begin(), nextMoves.end(), moveName) != nextMoves.end();
}

float ComboManager::GetCancelWindow(const ComboContext& context) const {
    if (!context.currentMove || !context.currentComboNode) {
        return 0.0f;
    }
    
    float moveProgress = context.currentMoveTime / context.currentMove->GetTotalFrames();
    return (moveProgress >= context.currentComboNode->cancelWindowStart && 
            moveProgress <= context.currentComboNode->cancelWindowEnd) ? 1.0f : 0.0f;
}

void ComboManager::PlayMoveEffects(const Move& move, const FrameData& frameData, Core::Entity entityID) {
    // Play sound effects
    if (m_audioSystem && !frameData.sfxOnHit.empty()) {
        // m_audioSystem->PlaySound(frameData.sfxOnHit, entityID);
    }
    
    // Trigger visual effects
    if (!frameData.vfxOnHit.empty()) {
        // TODO: Trigger VFX system
    }
}

void ComboManager::ApplyHitstop(ComboContext& context, float duration) {
    context.hitstopTimer = duration;
    
    // Apply global hitstop through GameFeelSystem
    if (m_gameFeelSystem) {
        m_gameFeelSystem->ApplyHitStop(duration);
    }
}

void ComboManager::TriggerScreenShake(float intensity) {
    if (m_gameFeelSystem) {
        // Use punch shake for combat hits - sharp and impactful
        m_gameFeelSystem->TriggerPunchShake(intensity, 0.15f);
    } else {
        std::cout << "Screen shake: " << intensity << std::endl;
    }
}

const ComboContext* ComboManager::GetComboContext(uint32_t entityId) const {
    auto it = m_comboContexts.find(entityId);
    return (it != m_comboContexts.end()) ? &(it->second) : nullptr;
}

} // namespace Combat
} // namespace CudaGame
