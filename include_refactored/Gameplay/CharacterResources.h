#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "Animation/AnimationSystem.h" // For AnimationState enum

namespace CudaGame {
namespace Gameplay {

// Forward declarations
namespace Assets {
    struct MeshAsset; // Placeholder for asset manager integration
}

/**
 * [AAA Pattern] Resource ID / Hash
 * In a real engine, we'd use hashed strings (uint32_t) for fast lookups.
 * For now, std::string is acceptable for readability but typed for future optimization.
 */
using ResourceID = std::string;

/**
 * [AAA Pattern] Animation Set Definition
 * Defines a reusable collection of animations for a specific archetype.
 * Example: "Knight_AnimSet" maps IDLE -> "Knight_Idle.fbx"
 * Data-Driven: Can be serialized to JSON.
 */
struct AnimationSet {
    ResourceID name;
    
    // Mapping from gameplay state to animation clip resource ID
    std::unordered_map<Animation::AnimationState, ResourceID> stateToClip;
    
    // Optional: Layered animations (Upper Body, Lower Body)
    // std::unordered_map<AnimationState, ResourceID> upperBodyClips;
};

/**
 * [AAA Pattern] Character Profile (Archetype Definition)
 * Defines the static properties of a character type.
 * Instances of this profile create Entities in the world.
 */
struct CharacterProfile {
    ResourceID profileName; // e.g. "RoyalGuard"
    
    // Visual Assets
    ResourceID meshID;      // e.g. "models/knight.fbx"
    ResourceID skeletonID;  // e.g. "skeletons/humanoid_v1.json" (or implicitly part of mesh)
    ResourceID animSetID;   // e.g. "animsets/knight_sword_shield.json"
    
    // Gameplay Stats (Base values)
    float baseHealth = 100.0f;
    float walkSpeed = 4.0f;
    float runSpeed = 8.0f;
    float mass = 80.0f;
    
    // Physics Configuration
    float colliderRadius = 0.5f;
    float colliderHeight = 1.8f;
    
    // Combat
    ResourceID startingWeaponID;
    ResourceID aiBehaviorID; // e.g. "PatrolAndChase"
    
    // Tags / Faction
    std::string faction = "Neutral";
    std::vector<std::string> tags;
};

} // namespace Gameplay
} // namespace CudaGame
