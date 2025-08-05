#include "AnimationSystem.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace CudaGame {
namespace Animation {

// RhythmAnalyzer Implementation
RhythmAnalyzer::RhythmAnalyzer() 
    : m_bpm(120.0f), m_currentPhase(0.0f), m_beatConfidence(0.0f), 
      m_adaptiveThreshold(0.5f), m_isCalibrated(false) {
    
    // Initialize frequency bins for beat detection
    m_frequencyBins.resize(32, 0.0f);
    m_energyHistory.resize(ENERGY_HISTORY_SIZE, 0.0f);
    m_beatHistory.resize(BEAT_HISTORY_SIZE, false);
    
    std::cout << "🎵 Rhythm Analyzer initialized with " << m_bpm << " BPM baseline" << std::endl;
}

void RhythmAnalyzer::initialize() {
    std::cout << "🎼 Initializing Rhythm Analysis System..." << std::endl;
    
    // Initialize audio processing buffers
    m_audioBuffer.resize(AUDIO_BUFFER_SIZE, 0.0f);
    m_fftBuffer.resize(FFT_SIZE, 0.0f);
    
    // Initialize beat detection parameters
    m_beatDetector.sensitivity = 0.7f;
    m_beatDetector.minInterval = 0.2f; // Minimum 200ms between beats
    m_beatDetector.maxInterval = 2.0f; // Maximum 2s between beats
    m_beatDetector.lastBeatTime = 0.0f;
    
    // Initialize adaptive parameters
    m_adaptiveParams.learningRate = 0.1f;
    m_adaptiveParams.confidenceThreshold = 0.6f;
    m_adaptiveParams.stabilityWindow = 8; // 8 beats for stability
    
    std::cout << "✅ Rhythm Analysis System ready" << std::endl;
}

void RhythmAnalyzer::processAudioFrame(const std::vector<float>& audioData, float deltaTime) {
    // Update timing
    m_currentTime += deltaTime;
    
    // Process audio data for beat detection
    if (!audioData.empty()) {
        updateAudioBuffer(audioData);
        performFrequencyAnalysis();
        detectBeats(deltaTime);
    }
    
    // Update rhythm phase based on current BPM
    updateRhythmPhase(deltaTime);
    
    // Adaptive BPM adjustment
    if (m_isCalibrated) {
        adaptiveBPMUpdate();
    }
    
    // Update beat confidence
    updateBeatConfidence();
}

void RhythmAnalyzer::updateAudioBuffer(const std::vector<float>& newData) {
    size_t copySize = std::min(newData.size(), m_audioBuffer.size());
    
    // Shift existing data left
    std::rotate(m_audioBuffer.begin(), m_audioBuffer.begin() + copySize, m_audioBuffer.end());
    
    // Copy new data to end
    std::copy(newData.begin(), newData.begin() + copySize, 
              m_audioBuffer.end() - copySize);
}

void RhythmAnalyzer::performFrequencyAnalysis() {
    // Copy audio buffer to FFT buffer
    std::copy(m_audioBuffer.end() - FFT_SIZE, m_audioBuffer.end(), m_fftBuffer.begin());
    
    // Apply window function (Hanning window)
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        float window = 0.5f * (1.0f - cos(2.0f * M_PI * i / (FFT_SIZE - 1)));
        m_fftBuffer[i] *= window;
    }
    
    // Perform simplified FFT (for demo - real implementation would use proper FFT)
    performSimplifiedFFT();
    
    // Extract frequency bins for beat detection
    extractFrequencyBins();
}

void RhythmAnalyzer::performSimplifiedFFT() {
    // Simplified frequency domain analysis
    // In real implementation, use FFTW or similar library
    
    for (size_t bin = 0; bin < m_frequencyBins.size(); ++bin) {
        float frequency = (float)bin * 22050.0f / m_frequencyBins.size(); // Nyquist at 22.05kHz
        float magnitude = 0.0f;
        
        // Calculate magnitude for this frequency bin
        for (size_t i = 0; i < FFT_SIZE; ++i) {
            float phase = 2.0f * M_PI * frequency * i / 44100.0f; // 44.1kHz sample rate
            float real = m_fftBuffer[i] * cos(phase);
            float imag = m_fftBuffer[i] * sin(phase);
            magnitude += sqrt(real * real + imag * imag);
        }
        
        m_frequencyBins[bin] = magnitude / FFT_SIZE;
    }
}

void RhythmAnalyzer::extractFrequencyBins() {
    // Focus on bass frequencies for beat detection (20Hz - 250Hz)
    float bassEnergy = 0.0f;
    for (size_t i = 0; i < 8; ++i) { // First 8 bins cover bass range
        bassEnergy += m_frequencyBins[i];
    }
    
    // Focus on mid frequencies for rhythm patterns (250Hz - 4kHz)
    float midEnergy = 0.0f;
    for (size_t i = 8; i < 24; ++i) {
        midEnergy += m_frequencyBins[i];
    }
    
    // Calculate total energy
    float totalEnergy = bassEnergy + midEnergy;
    
    // Update energy history
    m_energyHistory.push_back(totalEnergy);
    if (m_energyHistory.size() > ENERGY_HISTORY_SIZE) {
        m_energyHistory.erase(m_energyHistory.begin());
    }
    
    // Store bass energy ratio for beat detection
    m_currentBassRatio = (totalEnergy > 0.0f) ? (bassEnergy / totalEnergy) : 0.0f;
}

void RhythmAnalyzer::detectBeats(float deltaTime) {
    if (m_energyHistory.size() < 10) return; // Need some history
    
    // Calculate energy variance for beat detection
    float currentEnergy = m_energyHistory.back();
    float averageEnergy = 0.0f;
    
    for (float energy : m_energyHistory) {
        averageEnergy += energy;
    }
    averageEnergy /= m_energyHistory.size();
    
    // Beat detection using energy spike analysis
    float energyRatio = (averageEnergy > 0.0f) ? (currentEnergy / averageEnergy) : 0.0f;
    
    bool beatDetected = false;
    
    // Multiple beat detection criteria
    if (energyRatio > m_adaptiveThreshold && 
        (m_currentTime - m_beatDetector.lastBeatTime) > m_beatDetector.minInterval) {
        
        // Additional validation using bass energy
        if (m_currentBassRatio > 0.3f) {
            beatDetected = true;
            m_beatDetector.lastBeatTime = m_currentTime;
            
            // Update BPM based on beat interval
            updateDetectedBPM(deltaTime);
            
            std::cout << "🥁 Beat detected! Energy ratio: " << energyRatio 
                      << ", Bass ratio: " << m_currentBassRatio << std::endl;
        }
    }
    
    // Update beat history
    m_beatHistory.push_back(beatDetected);
    if (m_beatHistory.size() > BEAT_HISTORY_SIZE) {
        m_beatHistory.erase(m_beatHistory.begin());
    }
    
    m_currentBeatDetected = beatDetected;
}

void RhythmAnalyzer::updateDetectedBPM(float beatInterval) {
    if (beatInterval > 0.0f) {
        float detectedBPM = 60.0f / beatInterval;
        
        // Validate BPM range (60-200 BPM)
        if (detectedBPM >= 60.0f && detectedBPM <= 200.0f) {
            // Smooth BPM changes
            float smoothingFactor = m_adaptiveParams.learningRate;
            m_detectedBPM = m_detectedBPM * (1.0f - smoothingFactor) + detectedBPM * smoothingFactor;
            
            // Update confidence based on consistency
            float bpmDifference = abs(m_detectedBPM - m_bpm);
            if (bpmDifference < 5.0f) {
                m_beatConfidence = std::min(1.0f, m_beatConfidence + 0.1f);
            }
        }
    }
}

void RhythmAnalyzer::updateRhythmPhase(float deltaTime) {
    // Calculate phase based on current BPM
    float beatsPerSecond = m_bpm / 60.0f;
    float phaseIncrement = beatsPerSecond * deltaTime;
    
    m_currentPhase += phaseIncrement;
    
    // Wrap phase to [0, 1]
    while (m_currentPhase >= 1.0f) {
        m_currentPhase -= 1.0f;
    }
    
    // Check if we're on a beat (phase near 0)
    m_isOnBeat = (m_currentPhase < 0.1f) || (m_currentPhase > 0.9f);
    
    // Calculate beat intensity based on phase
    m_beatIntensity = calculateBeatIntensity();
}

float RhythmAnalyzer::calculateBeatIntensity() {
    // Create intensity curve that peaks at beats
    float intensity = 0.0f;
    
    if (m_currentPhase < 0.5f) {
        // Rising intensity towards beat
        intensity = 1.0f - (m_currentPhase * 2.0f);
    } else {
        // Falling intensity after beat
        intensity = (m_currentPhase - 0.5f) * 2.0f;
    }
    
    // Apply confidence multiplier
    intensity *= m_beatConfidence;
    
    // Add detection boost
    if (m_currentBeatDetected) {
        intensity = std::min(1.0f, intensity + 0.3f);
    }
    
    return intensity;
}

void RhythmAnalyzer::adaptiveBPMUpdate() {
    if (!m_isCalibrated) return;
    
    // Check if detected BPM is consistent
    if (m_beatConfidence > m_adaptiveParams.confidenceThreshold) {
        float bpmDifference = abs(m_detectedBPM - m_bpm);
        
        // Gradually adjust BPM if detection is confident and consistent
        if (bpmDifference > 2.0f) {
            float adjustmentRate = m_adaptiveParams.learningRate * 0.5f; // Slower for BPM changes
            m_bpm = m_bpm * (1.0f - adjustmentRate) + m_detectedBPM * adjustmentRate;
            
            std::cout << "🎯 Adaptive BPM update: " << m_bpm << " (detected: " << m_detectedBPM << ")" << std::endl;
        }
    }
}

void RhythmAnalyzer::updateBeatConfidence() {
    // Calculate confidence based on beat detection consistency
    int recentBeats = 0;
    size_t checkRange = std::min((size_t)m_adaptiveParams.stabilityWindow, m_beatHistory.size());
    
    for (size_t i = m_beatHistory.size() - checkRange; i < m_beatHistory.size(); ++i) {
        if (m_beatHistory[i]) recentBeats++;
    }
    
    // Expected beats in the window
    float expectedBeats = (m_bpm / 60.0f) * checkRange * (1.0f / 60.0f); // Assuming 60 FPS
    float beatAccuracy = (expectedBeats > 0.0f) ? (recentBeats / expectedBeats) : 0.0f;
    
    // Update confidence
    float targetConfidence = std::clamp(beatAccuracy, 0.0f, 1.0f);
    m_beatConfidence = m_beatConfidence * 0.9f + targetConfidence * 0.1f;
}

void RhythmAnalyzer::calibrate(float knownBPM) {
    m_bpm = knownBPM;
    m_detectedBPM = knownBPM;
    m_isCalibrated = true;
    m_beatConfidence = 0.8f; // Start with high confidence for known BPM
    
    std::cout << "🎼 Rhythm Analyzer calibrated to " << knownBPM << " BPM" << std::endl;
}

// RhythmFeedbackSystem Implementation
RhythmFeedbackSystem::RhythmFeedbackSystem() 
    : m_syncStrength(1.0f), m_feedbackIntensity(0.8f), m_isActive(true) {
    
    // Initialize feedback modifiers
    m_movementModifiers.speedMultiplier = 1.0f;
    m_movementModifiers.animationSpeedMultiplier = 1.0f;
    m_movementModifiers.responseAmplification = 1.0f;
    
    // Initialize visual effects parameters
    m_visualEffects.particleIntensity = 1.0f;
    m_visualEffects.colorSaturation = 1.0f;
    m_visualEffects.pulseAmplitude = 0.1f;
    
    std::cout << "🎨 Rhythm Feedback System initialized" << std::endl;
}

void RhythmFeedbackSystem::initialize(AnimationController* animationController) {
    m_animationController = animationController;
    
    // Initialize rhythm analyzer
    m_rhythmAnalyzer.initialize();
    
    // Setup default sync parameters
    m_syncParameters.beatSyncEnabled = true;
    m_syncParameters.phaseSyncEnabled = true;
    m_syncParameters.intensitySyncEnabled = true;
    m_syncParameters.adaptiveSync = true;
    
    // Initialize effect layers
    initializeEffectLayers();
    
    std::cout << "✅ Rhythm Feedback System ready with Animation Controller link" << std::endl;
}

void RhythmFeedbackSystem::initializeEffectLayers() {
    // Movement enhancement layer
    RhythmEffectLayer movementLayer;
    movementLayer.name = "movement";
    movementLayer.weight = 1.0f;
    movementLayer.type = RhythmEffectType::MOVEMENT_ENHANCEMENT;
    movementLayer.enabled = true;
    m_effectLayers.push_back(movementLayer);
    
    // Visual pulse layer
    RhythmEffectLayer visualLayer;
    visualLayer.name = "visual_pulse";
    visualLayer.weight = 0.8f;
    visualLayer.type = RhythmEffectType::VISUAL_PULSE;
    visualLayer.enabled = true;
    m_effectLayers.push_back(visualLayer);
    
    // Audio response layer
    RhythmEffectLayer audioLayer;
    audioLayer.name = "audio_response";
    audioLayer.weight = 0.6f;
    audioLayer.type = RhythmEffectType::AUDIO_RESPONSE;
    audioLayer.enabled = true;
    m_effectLayers.push_back(audioLayer);
    
    // Haptic feedback layer
    RhythmEffectLayer hapticLayer;
    hapticLayer.name = "haptic";
    hapticLayer.weight = 0.4f;
    hapticLayer.type = RhythmEffectType::HAPTIC_FEEDBACK;
    hapticLayer.enabled = false; // Disabled by default
    m_effectLayers.push_back(hapticLayer);
}

void RhythmFeedbackSystem::update(const std::vector<float>& audioData, float deltaTime) {
    if (!m_isActive) return;
    
    // Update rhythm analysis
    m_rhythmAnalyzer.processAudioFrame(audioData, deltaTime);
    
    // Get rhythm parameters
    RhythmParameters rhythmParams = m_rhythmAnalyzer.getRhythmParameters();
    
    // Update animation controller with rhythm data
    if (m_animationController) {
        m_animationController->setRhythmParameters(
            rhythmParams.phase,
            rhythmParams.isOnBeat,
            rhythmParams.intensity
        );
    }
    
    // Apply rhythm feedback effects
    applyRhythmEffects(rhythmParams, deltaTime);
    
    // Update movement modifiers
    updateMovementModifiers(rhythmParams);
    
    // Update visual effects
    updateVisualEffects(rhythmParams);
    
    // Trigger rhythm events
    triggerRhythmEvents(rhythmParams);
}

void RhythmFeedbackSystem::applyRhythmEffects(const RhythmParameters& rhythmParams, float deltaTime) {
    for (auto& layer : m_effectLayers) {
        if (!layer.enabled) continue;
        
        float effectStrength = layer.weight * m_syncStrength * m_feedbackIntensity;
        
        switch (layer.type) {
            case RhythmEffectType::MOVEMENT_ENHANCEMENT:
                applyMovementEnhancement(rhythmParams, effectStrength);
                break;
                
            case RhythmEffectType::VISUAL_PULSE:
                applyVisualPulse(rhythmParams, effectStrength);
                break;
                
            case RhythmEffectType::AUDIO_RESPONSE:
                applyAudioResponse(rhythmParams, effectStrength);
                break;
                
            case RhythmEffectType::HAPTIC_FEEDBACK:
                applyHapticFeedback(rhythmParams, effectStrength);
                break;
        }
    }
}

void RhythmFeedbackSystem::applyMovementEnhancement(const RhythmParameters& rhythmParams, float strength) {
    // Enhance movement speed on beats
    if (rhythmParams.isOnBeat) {
        m_movementModifiers.speedMultiplier = 1.0f + (0.2f * strength * rhythmParams.intensity);
        m_movementModifiers.responseAmplification = 1.0f + (0.15f * strength);
    } else {
        // Gradually return to normal
        m_movementModifiers.speedMultiplier = lerp(m_movementModifiers.speedMultiplier, 1.0f, 0.1f);
        m_movementModifiers.responseAmplification = lerp(m_movementModifiers.responseAmplification, 1.0f, 0.1f);
    }
    
    // Sync animation speed to rhythm
    float phaseFactor = 1.0f + sin(rhythmParams.phase * 2.0f * M_PI) * 0.1f * strength;
    m_movementModifiers.animationSpeedMultiplier = phaseFactor;
}

void RhythmFeedbackSystem::applyVisualPulse(const RhythmParameters& rhythmParams, float strength) {
    // Create visual pulse effect
    float pulseIntensity = rhythmParams.intensity * strength;
    
    if (rhythmParams.isOnBeat) {
        m_visualEffects.pulseAmplitude = 0.2f * pulseIntensity;
        m_visualEffects.colorSaturation = 1.0f + (0.3f * pulseIntensity);
        m_visualEffects.particleIntensity = 1.0f + (0.5f * pulseIntensity);
    } else {
        // Phase-based modulation
        float phasePulse = sin(rhythmParams.phase * 2.0f * M_PI) * 0.1f * strength;
        m_visualEffects.pulseAmplitude = phasePulse;
        m_visualEffects.colorSaturation = 1.0f + phasePulse * 0.2f;
        m_visualEffects.particleIntensity = 1.0f + phasePulse * 0.3f;
    }
}

void RhythmFeedbackSystem::applyAudioResponse(const RhythmParameters& rhythmParams, float strength) {
    // Modulate audio parameters based on rhythm
    m_audioResponse.reverbWetness = 0.3f + (rhythmParams.intensity * 0.2f * strength);
    m_audioResponse.filterCutoff = 1000.0f + (sin(rhythmParams.phase * 2.0f * M_PI) * 500.0f * strength);
    
    if (rhythmParams.isOnBeat) {
        m_audioResponse.compressionRatio = 1.5f + (0.5f * strength);
    } else {
        m_audioResponse.compressionRatio = lerp(m_audioResponse.compressionRatio, 1.0f, 0.05f);
    }
}

void RhythmFeedbackSystem::applyHapticFeedback(const RhythmParameters& rhythmParams, float strength) {
    if (rhythmParams.isOnBeat) {
        // Generate haptic pulse on beat
        m_hapticFeedback.pulseStrength = rhythmParams.intensity * strength;
        m_hapticFeedback.pulseDuration = 0.1f;
        m_hapticFeedback.triggered = true;
        
        std::cout << "📳 Haptic beat pulse: " << m_hapticFeedback.pulseStrength << std::endl;
    }
}

void RhythmFeedbackSystem::updateMovementModifiers(const RhythmParameters& rhythmParams) {
    // Apply movement modifiers based on rhythm sync parameters
    if (m_syncParameters.beatSyncEnabled && rhythmParams.isOnBeat) {
        // Boost movement on beats
        m_currentMovementBoost = rhythmParams.intensity * m_syncStrength;
    } else {
        // Decay boost
        m_currentMovementBoost = lerp(m_currentMovementBoost, 0.0f, 0.2f);
    }
    
    if (m_syncParameters.phaseSyncEnabled) {
        // Continuous phase-based modulation
        float phaseModulation = sin(rhythmParams.phase * 2.0f * M_PI) * 0.1f * m_syncStrength;
        m_currentPhaseModulation = phaseModulation;
    }
}

void RhythmFeedbackSystem::updateVisualEffects(const RhythmParameters& rhythmParams) {
    // Update visual effect parameters
    m_visualEffects.currentPhase = rhythmParams.phase;
    m_visualEffects.currentIntensity = rhythmParams.intensity;
    m_visualEffects.isOnBeat = rhythmParams.isOnBeat;
    
    // Calculate screen shake intensity
    if (rhythmParams.isOnBeat) {
        m_visualEffects.screenShakeIntensity = rhythmParams.intensity * 0.5f * m_feedbackIntensity;
    } else {
        m_visualEffects.screenShakeIntensity = lerp(m_visualEffects.screenShakeIntensity, 0.0f, 0.3f);
    }
}

void RhythmFeedbackSystem::triggerRhythmEvents(const RhythmParameters& rhythmParams) {
    // Trigger events for other systems to respond to
    if (rhythmParams.isOnBeat && !m_previousBeatState) {
        // Beat start event
        RhythmEvent beatEvent;
        beatEvent.type = RhythmEventType::BEAT_HIT;
        beatEvent.intensity = rhythmParams.intensity;
        beatEvent.phase = rhythmParams.phase;
        beatEvent.bpm = m_rhythmAnalyzer.getBPM();
        
        m_rhythmEvents.push_back(beatEvent);
        
        std::cout << "🎵 Rhythm Beat Event - Intensity: " << rhythmParams.intensity 
                  << ", BPM: " << beatEvent.bpm << std::endl;
    }
    
    // Phase change events
    if (abs(rhythmParams.phase - m_previousPhase) > 0.25f) {
        RhythmEvent phaseEvent;
        phaseEvent.type = RhythmEventType::PHASE_CHANGE;
        phaseEvent.intensity = rhythmParams.intensity;
        phaseEvent.phase = rhythmParams.phase;
        phaseEvent.bpm = m_rhythmAnalyzer.getBPM();
        
        m_rhythmEvents.push_back(phaseEvent);
    }
    
    // Store previous state
    m_previousBeatState = rhythmParams.isOnBeat;
    m_previousPhase = rhythmParams.phase;
}

float RhythmFeedbackSystem::lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Getters for external systems
RhythmParameters RhythmFeedbackSystem::getRhythmParameters() const {
    return m_rhythmAnalyzer.getRhythmParameters();
}

MovementModifiers RhythmFeedbackSystem::getMovementModifiers() const {
    return m_movementModifiers;
}

VisualEffects RhythmFeedbackSystem::getVisualEffects() const {
    return m_visualEffects;
}

AudioResponse RhythmFeedbackSystem::getAudioResponse() const {
    return m_audioResponse;
}

std::vector<RhythmEvent> RhythmFeedbackSystem::consumeRhythmEvents() {
    std::vector<RhythmEvent> events = m_rhythmEvents;
    m_rhythmEvents.clear();
    return events;
}

// Configuration methods
void RhythmFeedbackSystem::setSyncStrength(float strength) {
    m_syncStrength = std::clamp(strength, 0.0f, 2.0f);
}

void RhythmFeedbackSystem::setFeedbackIntensity(float intensity) {
    m_feedbackIntensity = std::clamp(intensity, 0.0f, 1.0f);
}

void RhythmFeedbackSystem::enableEffectLayer(const std::string& layerName, bool enabled) {
    for (auto& layer : m_effectLayers) {
        if (layer.name == layerName) {
            layer.enabled = enabled;
            std::cout << "🎛️ Effect layer '" << layerName << "' " 
                      << (enabled ? "enabled" : "disabled") << std::endl;
            break;
        }
    }
}

void RhythmFeedbackSystem::setEffectLayerWeight(const std::string& layerName, float weight) {
    for (auto& layer : m_effectLayers) {
        if (layer.name == layerName) {
            layer.weight = std::clamp(weight, 0.0f, 1.0f);
            std::cout << "🎚️ Effect layer '" << layerName << "' weight set to " << weight << std::endl;
            break;
        }
    }
}

} // namespace Animation
} // namespace CudaGame
