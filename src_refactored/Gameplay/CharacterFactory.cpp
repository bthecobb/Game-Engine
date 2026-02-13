#include "Gameplay/CharacterFactory.h"
#include "Gameplay/PlayerComponents.h"
#include "Physics/PhysicsComponents.h"
#include "Physics/CharacterController.h"
#include "Animation/AnimationComponent.h"
#include "Gameplay/AnimationControllerComponent.h"
#include "AI/AIComponent.h"
#include "Rendering/RenderComponents.h"
#include "Rendering/DX12RenderPipeline.h" // For Mesh Creation
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

namespace CudaGame {
namespace Gameplay {

// ... (skip to line 89 context during apply? No, I must use valid context)
// I will split this into two replacements if needed, but I can't target line 89 easily with top include.
// Actually replace_file_content targets a BLOCK.
// I'll do two replace calls. One for includes, one for moveSpeed.

// This call is for INCLUDES only.

CharacterFactory::CharacterFactory() {
}

CharacterFactory::~CharacterFactory() {
    Shutdown();
}

bool CharacterFactory::Initialize() {
    std::cerr << "[CharacterFactory] Initialize ENTER" << std::endl;
    // Register Default Animation Sets
    AnimationSet warriorSet;
    warriorSet.name = "WarriorSet";
    warriorSet.stateToClip[Animation::AnimationState::IDLE] = "Wuson_Bind"; 
    warriorSet.stateToClip[Animation::AnimationState::WALKING] = "Wuson_Walk"; 
    warriorSet.stateToClip[Animation::AnimationState::RUNNING] = "Wuson_Run";
    warriorSet.stateToClip[Animation::AnimationState::JUMPING] = "Wuson_Run"; 
    std::cerr << "[CharacterFactory] Registering AnimationSet..." << std::endl;
    RegisterAnimationSet("WarriorSet", warriorSet);
    
    // Register Default Profiles
    CharacterProfile knight;
    knight.profileName = "Knight";
    knight.meshID = "C:/Users/Brandon/CudaGame/assets/models/Testwuson.X"; 
    knight.animSetID = "WarriorSet";
    knight.skeletonID = "Humanoid"; 
    knight.baseHealth = 150.0f;
    knight.runSpeed = 5.0f;
    std::cerr << "[CharacterFactory] Registering Profile Knight..." << std::endl;
    RegisterProfile("Knight", knight);
    
    CharacterProfile procTest;
    procTest.profileName = "ProceduralTest";
    procTest.meshID = "internal:ProceduralArm";
    procTest.animSetID = ""; 
    procTest.skeletonID = "Humanoid"; 
    procTest.baseHealth = 100.0f;
    procTest.runSpeed = 5.0f;
    std::cerr << "[CharacterFactory] Registering Profile ProceduralTest..." << std::endl;
    RegisterProfile("ProceduralTest", procTest);
    
    std::cerr << "[CharacterFactory] Creating Skeleton..." << std::endl;
    auto skeleton = std::make_shared<Animation::Skeleton>();
    
    Animation::Skeleton::Bone root;
    root.name = "Root";
    root.parentIndex = -1;
    root.localTransform = glm::mat4(1.0f);
    root.inverseBindPose = glm::mat4(1.0f);
    skeleton->bones.push_back(root);
    skeleton->boneNameToIndex["Root"] = 0;
    
    Animation::Skeleton::Bone spine;
    spine.name = "Spine";
    spine.parentIndex = 0;
    spine.localTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 1.0f, 0));
    spine.inverseBindPose = glm::inverse(spine.localTransform); 
    skeleton->bones.push_back(spine);
    skeleton->boneNameToIndex["Spine"] = 1;

    Animation::Skeleton::Bone head;
    head.name = "Head";
    head.parentIndex = 1;
    head.localTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0.5f, 0));
    head.inverseBindPose = glm::inverse(root.localTransform * spine.localTransform * head.localTransform); 
    skeleton->bones.push_back(head);
    skeleton->boneNameToIndex["Head"] = 2;

    std::cerr << "[CharacterFactory] Registering Skeleton..." << std::endl;
    RegisterSkeleton("Humanoid", skeleton);
    
    std::cerr << "[CharacterFactory] Initialize EXIT Success" << std::endl;
    return true;
}

void CharacterFactory::RegisterProfile(const std::string& profileName, const CharacterProfile& profile) {
    m_profiles[profileName] = profile;
}

void CharacterFactory::RegisterAnimationSet(const std::string& setName, const AnimationSet& animSet) {
    m_animationSets[setName] = animSet;
}

void CharacterFactory::RegisterSkeleton(const std::string& skeletonID, std::shared_ptr<Animation::Skeleton> skeleton) {
    m_skeletons[skeletonID] = skeleton;
}

void CharacterFactory::RegisterWeaponDefinition(const std::string& weaponID, const WeaponDefinition& def) {
    m_weapons[weaponID] = def;
}

Core::Entity CharacterFactory::SpawnCharacter(const std::string& profileName, const glm::vec3& position) {
    std::cerr << "[CharacterFactory] Spawning " << profileName << "..." << std::endl;
    if (m_profiles.find(profileName) == m_profiles.end()) {
        std::cerr << "[CharacterFactory] Error: Profile '" << profileName << "' not found!" << std::endl;
        return 0; // Invalid Entity
    }
    
    const auto& profile = m_profiles[profileName];
    std::cerr << "[CharacterFactory] Profile found: " << profile.profileName << std::endl;
    
    auto& coordinator = Core::Coordinator::GetInstance();
    
    Core::Entity entity = coordinator.CreateEntity();
    std::cerr << "[CharacterFactory] Entity Created: " << entity << std::endl;
    
    // 1. Transform
    coordinator.AddComponent(entity, Rendering::TransformComponent{position, glm::vec3(0), glm::vec3(1)});
    std::cerr << "[CharacterFactory] Transform Added" << std::endl;
    
    // 2. Physics & Controller
    Physics::CharacterControllerComponent cct;
    cct.height = profile.colliderHeight;
    cct.radius = profile.colliderRadius;
    coordinator.AddComponent(entity, cct);
    
    Physics::RigidbodyComponent rb;
    rb.mass = profile.mass; 
    
    // Disable physics for Procedural Test (Kinematic) to prevent falling over
    if (profile.profileName == "ProceduralTest") {
        rb.useGravity = false;
        rb.isKinematic = true; 
    }
    
    coordinator.AddComponent(entity, rb);
    // std::cout << "[CharacterFactory] Rigidbody Added" << std::endl;

    // 2.5 Collider (Required for PhysX Actor creation)
    Physics::ColliderComponent collider;
    collider.shape = Physics::ColliderShape::CAPSULE;
    collider.radius = profile.colliderRadius;
    collider.halfExtents.y = profile.colliderHeight * 0.5f; // PhysXSystem uses halfExtents.y for capsule halfHeight
    coordinator.AddComponent(entity, collider);
    // std::cout << "[CharacterFactory] Collider Added" << std::endl;
    
    // 3. Gameplay Stats
    PlayerMovementComponent mv;
    mv.baseSpeed = profile.runSpeed; // Map from profile
    coordinator.AddComponent(entity, mv);
    // std::cout << "[CharacterFactory] Movement Added" << std::endl;
    
    // 4. Animation
    Animation::AnimationComponent animComp;
    // Skeleton should be loaded from profile.skeletonID.
    if (!profile.skeletonID.empty() && m_skeletons.count(profile.skeletonID)) {
        animComp.skeleton = m_skeletons[profile.skeletonID];
    } else {
        std::cerr << "[CharacterFactory] Warning: Skeleton ID '" << profile.skeletonID << "' not found or empty." << std::endl;
    }

    // Load Animation Set (State Mapping)
    if (!profile.animSetID.empty() && m_animationSets.count(profile.animSetID)) {
        animComp.stateMap = m_animationSets[profile.animSetID].stateToClip;
    }

    coordinator.AddComponent(entity, animComp);
    // std::cout << "[CharacterFactory] AnimationComponent Added" << std::endl;
    
    // 5. Logic (Animation Controller)
    AnimationControllerComponent animCtrl;
    animCtrl.animationSetID = profile.animSetID;
    coordinator.AddComponent(entity, animCtrl);
    // std::cout << "[CharacterFactory] Controller Added" << std::endl;
    
    // 6. Visuals (Mesh)
    Rendering::MeshComponent meshComp;
    meshComp.modelPath = profile.meshID;
    coordinator.AddComponent(entity, meshComp);
    std::cout << "[CharacterFactory] MeshComponent Added: " << meshComp.modelPath << std::endl;
    
    // Setup Transform with Scale correction for imported mesh (0.5f = 5x previous size)
    // coordinator.AddComponent(entity, Rendering::TransformComponent{position, glm::vec3(0), glm::vec3(0.5f)});
    // NOTE: Transform already added at start. We should modify existing component if needed, not Add again.
    auto& trans = coordinator.GetComponent<Rendering::TransformComponent>(entity);
    trans.scale = glm::vec3(0.5f);

    // Diagnostic Blue Material
    Rendering::MaterialComponent mat;
    mat.albedo = glm::vec3(0.0f, 0.0f, 1.0f); // Pure Blue
    mat.metallic = 0.5f;
    mat.roughness = 0.5f;
    coordinator.AddComponent(entity, mat);
    
    std::cout << "[CharacterFactory] Material (Blue) & Mesh Added (" << profile.meshID << ")" << std::endl;
    
    // 7. Combat Setup (Phase 6)
    CombatComponent combat;
    combat.health = profile.baseHealth;
    combat.maxHealth = profile.baseHealth;
    combat.faction = profile.faction;
    coordinator.AddComponent(entity, combat);
    std::cout << "[CharacterFactory] Combat Added" << std::endl;
    
    // 8. AI Setup (Phase 7)
    if (!profile.aiBehaviorID.empty()) {
        AI::AIComponent aiComp;
        
        // Construct simple tree (Mocking a script loader)
        if (profile.aiBehaviorID == "ChasePlayer") {
            auto root = std::make_shared<AI::Sequence>();
            aiComp.rootNode = root;
        }
        
        coordinator.AddComponent(entity, aiComp);
        
        // AI needs Input Component to drive
        coordinator.AddComponent(entity, PlayerInputComponent{});
    }
    std::cout << "[CharacterFactory] AI Added" << std::endl;
    
    // Spawn Weapon if defined
    if (!profile.startingWeaponID.empty()) {
        SpawnWeapon(entity, profile.startingWeaponID);
    }
    std::cout << "[CharacterFactory] Weapons Processed" << std::endl;
    
    std::cout << "[CharacterFactory] Spawned " << profileName << " at " << position.x << ", " << position.y << ", " << position.z << " (Entity " << entity << ")" << std::endl;
    
    return entity;
}

void CharacterFactory::SpawnWeapon(Core::Entity owner, const std::string& weaponID) {
    if (m_weapons.find(weaponID) == m_weapons.end()) {
        std::cerr << "[CharacterFactory] Error: Weapon '" << weaponID << "' not found!" << std::endl;
        return;
    }
    
    const auto& def = m_weapons[weaponID];
    auto& coordinator = Core::Coordinator::GetInstance();
    
    // Create Weapon Entity
    Core::Entity weaponEntity = coordinator.CreateEntity();
    
    // 1. Transform (Child of Owner - ideally attached to Hand Bone)
    // For now, simple parenting or position sync
    Rendering::TransformComponent transform;
    transform.scale = glm::vec3(1.0f);
    // Add parenting component if we had one, or handle in Update
    coordinator.AddComponent(weaponEntity, transform);
    
    // 2. Weapon State
    WeaponComponent weaponComp;
    weaponComp.definitionID = weaponID;
    weaponComp.owner = owner;
    weaponComp.currentAmmo = def.maxAmmo;
    coordinator.AddComponent(weaponEntity, weaponComp);
    
    // 3. Visuals
    // Ideally def.meshID determines the mesh.
    // For verification, we add a Material.
    coordinator.AddComponent(weaponEntity, Rendering::MaterialComponent{glm::vec4(1,0,0,1), 0.8f, 0.5f}); // Red weapon
    
    // 4. Link to Owner
    auto& combat = coordinator.GetComponent<CombatComponent>(owner);
    combat.activeWeaponEntity = weaponEntity;
    combat.inventory.push_back(weaponEntity);
    
    std::cout << "[CharacterFactory] Spawned Weapon '" << weaponID << "' for Entity " << owner << std::endl;
}

// SetupPlayer removed

void CharacterFactory::AssembleCharacter(Core::Entity entity, const CharacterProfile& profile) {
    (void)entity;
    (void)profile;
    // Helper used by SpawnCharacter
}

void CharacterFactory::Shutdown() {
    m_profiles.clear();
    m_animationSets.clear();
}

} // namespace Gameplay
} // namespace CudaGame
