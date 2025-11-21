# Full Game Integration Plan - D3D12 + PhysX + ECS

## Status: In Progress
**Completed**: Gameplay component registration  
**Current Phase**: System integration  
**Target**: Playable combat game with rhythm mechanics

---

## Overview

We're integrating the original CudaGame systems (combat, movement, enemies, rhythm) with the new AAA-quality D3D12 rendering pipeline and PhysX physics. The game is a **3D action game with rhythm-based combat** featuring:

- **Player Movement**: WASD, dashing, wall-running, double-jump
- **Combat System**: Frame-perfect attacks, combos, parrying, weapon switching (Sword/Staff/Hammer)
- **Rhythm Integration**: Timing attacks to beats for damage multipliers
- **Enemy AI**: Patrol, chase, attack with line-of-sight detection
- **Physics-Based**: All movement and combat use PhysX rigidbodies

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Game Loop                              │
│  (Full3DGame_DX12.cpp main())                                │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌───────▼────────┐   ┌───────▼────────┐
│   Input        │   │  Game Systems  │   │   Rendering    │
│   (GLFW)       │   │   (ECS)        │   │   (D3D12)      │
├────────────────┤   ├────────────────┤   ├────────────────┤
│• WASD          │   │• Movement      │   │• Deferred      │
│• Mouse         │   │• Combat        │   │• G-Buffer      │
│• Keyboard      │   │• AI            │   │• Lighting      │
└────────────────┘   │• Rhythm        │   │• Post-process  │
                     │• Weapons       │   └────────────────┘
                     └────────────────┘
                              │
                     ┌────────▼────────┐
                     │    Physics      │
                     │    (PhysX)      │
                     ├─────────────────┤
                     │• Forces/Impulses│
                     │• Collision Det. │
                     │• Raycasts       │
                     └─────────────────┘
                              │
                     ┌────────▼────────┐
                     │      ECS        │
                     │  (Components)   │
                     ├─────────────────┤
                     │• Transform      │
                     │• Material       │
                     │• Player*        │
                     │• Enemy*         │
                     │• Combat         │
                     └─────────────────┘
```

---

## Phase 1: Core Systems Integration (CURRENT)

### ✅ Completed
- [x] PhysX physics engine integrated
- [x] D3D12 deferred rendering pipeline
- [x] ECS component registration (all gameplay components)
- [x] Basic WASD movement with physics forces
- [x] Camera follow system
- [x] Physics-ECS-Render sync loop

### 🚧 In Progress

#### 1.1 Enhanced Player Movement System
**File**: `src/Gameplay/PlayerMovementSystem.cpp`

**Current State**: Basic WASD + jump via direct physics forces in main loop

**Target State**: Full-featured PlayerMovementSystem with:
- **Dash**: SHIFT key, 80N force for 0.2s, 1s cooldown
- **Sprint**: Hold SHIFT while moving, 2x speed multiplier
- **Double Jump**: Space in air, can only jump once
- **Wall Running**: Auto-detect walls, 25 units/s parallel movement, 2s duration
- **Momentum**: Gradual acceleration/deceleration curves
- **Grounded Check**: Raycast down 0.1 units for proper jump control

**Implementation Plan**:
1. Move input handling from main() to PlayerMovementSystem::Update()
2. Read PlayerInputComponent (populate from GLFW in main)
3. Write to PlayerMovementComponent (velocity, state, timers)
4. Apply forces/impulses to PhysX actor based on component state
5. Update MovementState enum (IDLE → WALKING → RUNNING → DASHING)

**Key Challenges**:
- Grounded detection: PhysX raycast vs. velocity check
- Wall running: PhysX overlap query for nearby walls
- Dash cooldown: Track in PlayerMovementComponent.dashCooldownTimer

---

#### 1.2 Combat System Integration
**File**: `src/Combat/CombatSystem.cpp` (existing)

**Current State**: System implemented but not connected to game loop

**Target State**: Working combat with:
- **Left Mouse**: Light attack (punch/weapon attack)
- **Right Mouse**: Heavy attack (stronger, slower)
- **Mouse4**: Block/Parry
- **Number Keys (1-3)**: Switch weapons (Sword/Staff/Hammer)
- **Combo System**: Chain attacks within 0.8s window
- **Hit Detection**: PhysX sphere overlap at attack range
- **Damage**: Apply to EnemyCombatComponent.health

**Implementation Plan**:
1. Instantiate CombatSystem in main() after physicsSystem
2. Call combatSystem->Update(deltaTime) in game loop
3. Pass input to processCombatInput(playerEntity, inputFlags)
4. Convert mouse buttons to input flags (ATTACK_LIGHT, ATTACK_HEAVY, BLOCK)
5. Implement hit detection using PhysX PxScene::overlap()
6. Apply damage to enemy health components
7. Update PlayerCombatComponent.combatState

**Key Challenges**:
- PhysX overlap query: Create PxSphereGeometry at player forward + attackRange
- Combo timing: Use combatTimer and comboWindow from CombatComponent
- Animation sync: Stub for now (just update combatState)

---

#### 1.3 Input System Refactor
**Current**: Direct GLFW polling in main loop  
**Target**: Centralized InputSystem that populates PlayerInputComponent

**Implementation**:
```cpp
void UpdateInputComponent(Core::Entity playerEntity) {
    auto& input = coordinator.GetComponent<PlayerInputComponent>(playerEntity);
    
    // Keyboard
    for (int i = 0; i < 1024; ++i) {
        input.keys[i] = (glfwGetKey(window, i) == GLFW_PRESS);
    }
    
    // Mouse buttons
    for (int i = 0; i < 8; ++i) {
        input.mouseButtons[i] = (glfwGetMouseButton(window, i) == GLFW_PRESS);
    }
    
    // Mouse position (for aiming)
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    glm::vec2 newPos(xpos, ypos);
    input.mouseDelta = newPos - input.mousePos;
    input.mousePos = newPos;
}
```

Place before PlayerMovementSystem and CombatSystem updates.

---

## Phase 2: Enemy AI & Combat (NEXT)

### 2.1 Enemy Spawning
**Where**: Extend `CreateGameWorld()` in Full3DGame_DX12.cpp

**Spawn**: 3-5 enemies around player at radius 10-20 units

**Each Enemy Needs**:
- TransformComponent (position, rotation, scale)
- MaterialComponent (red color to distinguish from player)
- PhysX capsule rigidbody (similar to player)
- EnemyAIComponent (AIState::PATROL)
- EnemyCombatComponent (health: 75, damage: 15)
- EnemyMovementComponent (speed: 10)
- TargetingComponent (target player)

**Physics Setup**:
```cpp
PxRigidDynamic* enemyActor = physics->createRigidDynamic(PxTransform(...));
PxShape* enemyShape = physics->createShape(PxCapsuleGeometry(0.4f, 0.6f), *material);
enemyActor->attachShape(*enemyShape);
PxRigidBodyExt::updateMassAndInertia(*enemyActor, 70.0f); // 70kg
// Lock rotations like player
enemyActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X, true);
enemyActor->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z, true);
scene->addActor(*enemyActor);
```

---

### 2.2 EnemyAISystem
**File**: Create `src/Gameplay/EnemyAISystem.cpp`

**AI States**:
1. **PATROL**: Move between patrol points, wait at each
2. **CHASE**: Player detected, move toward lastKnownPlayerPos
3. **ATTACK**: Within attackRange (3.5 units), trigger attack
4. **STUNNED**: Take damage, cannot act for 1s
5. **RETREATING**: Low health (<25%), move away from player

**Detection Logic** (every frame):
```cpp
glm::vec3 toPlayer = playerPos - enemyPos;
float distance = glm::length(toPlayer);

// Range check
if (distance > detectionRange) return; // Too far

// Vision cone check (60° FOV)
glm::vec3 forward = glm::normalize(enemyForward);
glm::vec3 dirToPlayer = glm::normalize(toPlayer);
float dot = glm::dot(forward, dirToPlayer);
float angle = glm::degrees(acos(dot));

if (angle < visionAngle / 2.0f) {
    // In FOV, now check line-of-sight with PhysX raycast
    PxRaycastBuffer hit;
    bool blocked = scene->raycast(enemyPos, dirToPlayer, distance, hit);
    if (!blocked || hit.block.actor == playerActor) {
        // Can see player!
        aiComponent.aiState = AIState::CHASE;
        aiComponent.lastKnownPlayerPos = playerPos;
    }
}
```

**Movement**: Apply forces to enemy PhysX actor
```cpp
glm::vec3 targetDir = glm::normalize(targetPos - enemyPos);
PxVec3 force(targetDir.x * speed * 100.0f, 0.0f, targetDir.z * speed * 100.0f);
enemyActor->addForce(force);
```

---

### 2.3 Enemy Attacks
**Trigger**: EnemyAISystem calls `combatSystem->attemptAttack(enemyEntity, playerEntity)`

**Hit Detection**: PhysX sphere overlap at enemy forward + attackRange

**Damage Application**:
```cpp
auto& playerCombat = coordinator.GetComponent<PlayerCombatComponent>(playerEntity);
playerCombat.health -= enemyDamage;
if (playerCombat.health <= 0) {
    // Player died - trigger game over state
}
```

---

## Phase 3: Weapons & Inventory

### 3.1 Weapon Data
**Already Defined**: `include/Combat/CombatSystem.h` has WeaponType enum and WeaponData struct

**Properties**:
- **SWORD**: Fast, 15 damage, 4m range, 3-hit combo
- **STAFF**: Magic, 12 damage, 6m range, slower, AOE
- **HAMMER**: Heavy, 25 damage, 3m range, slow, knockback

### 3.2 Weapon Switching
**Input**: Number keys 1, 2, 3  
**System**: CombatSystem::switchWeapon(playerEntity, newWeapon)

**Updates**:
- PlayerCombatComponent.currentWeapon
- PlayerCombatComponent.attackRange (from WeaponData)
- Material color (visual feedback):
  - Sword: Cyan (0.0, 0.8, 0.8)
  - Staff: Purple (0.6, 0.2, 0.8)
  - Hammer: Orange (0.9, 0.5, 0.1)

**Implementation**:
```cpp
if (input.keys[GLFW_KEY_1]) combatSystem->switchWeapon(playerEntity, WeaponType::SWORD);
if (input.keys[GLFW_KEY_2]) combatSystem->switchWeapon(playerEntity, WeaponType::STAFF);
if (input.keys[GLFW_KEY_3]) combatSystem->switchWeapon(playerEntity, WeaponType::HAMMER);
```

---

## Phase 4: Rhythm System

### 4.1 RhythmSystem
**File**: Create `src/Rhythm/RhythmSystem.cpp`

**Core Mechanic**: Generate beats at 120 BPM (0.5s intervals)

**Beat Timing**:
```cpp
class RhythmSystem {
    float beatInterval = 0.5f; // 120 BPM
    float beatTimer = 0.0f;
    float beatWindow = 0.1f; // ±0.1s = "good" timing
    float perfectWindow = 0.05f; // ±0.05s = "perfect" timing
    
    void Update(float deltaTime) {
        beatTimer += deltaTime;
        if (beatTimer >= beatInterval) {
            beatTimer -= beatInterval;
            OnBeat(); // Trigger beat event
        }
        
        // Check if we're near a beat
        float timeToBeat = beatInterval - beatTimer;
        bool isNearBeat = (timeToBeat < beatWindow) || (beatTimer < beatWindow);
        
        // Update all PlayerRhythmComponents
        for (auto entity : entitiesWithRhythm) {
            auto& rhythm = coordinator.GetComponent<PlayerRhythmComponent>(entity);
            rhythm.isOnBeat = isNearBeat;
            rhythm.beatTimer = beatTimer;
            
            // Calculate multiplier based on timing
            if (timeToBeat < perfectWindow || beatTimer < perfectWindow) {
                rhythm.rhythmMultiplier = 1.5f; // Perfect!
            } else if (isNearBeat) {
                rhythm.rhythmMultiplier = 1.25f; // Good
            } else {
                rhythm.rhythmMultiplier = 1.0f; // Normal
            }
        }
    }
};
```

### 4.2 Combat Integration
**When Player Attacks**:
```cpp
float finalDamage = baseDamage * rhythmMultiplier;
if (rhythmMultiplier > 1.2f) {
    // Visual feedback: screen flash, particle burst
    std::cout << "PERFECT HIT! (" << finalDamage << " damage)" << std::endl;
}
```

---

## Phase 5: UI & Polish

### 5.1 HUD Elements (Overlay)
**Health Bar**: Top-left, red bar
**Stamina/Energy**: Below health (for dash/special moves)
**Weapon Display**: Bottom-right, show current weapon icon
**Combo Counter**: Center-top, shows combo count + multiplier
**Rhythm Indicator**: Bottom-center, pulsing circle that expands on beat

### 5.2 ImGui Integration (Quick Path)
Instead of D3D12 2D rendering, use ImGui for rapid prototyping:

**Add to CMakeLists.txt**:
```cmake
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.0
)
FetchContent_MakeAvailable(imgui)
```

**Render HUD**:
```cpp
void RenderHUD() {
    auto& playerCombat = coordinator.GetComponent<PlayerCombatComponent>(playerEntity);
    auto& playerRhythm = coordinator.GetComponent<PlayerRhythmComponent>(playerEntity);
    
    ImGui::Begin("HUD", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground);
    
    // Health
    ImGui::Text("Health: %.0f / %.0f", playerCombat.health, playerCombat.maxHealth);
    ImGui::ProgressBar(playerCombat.health / playerCombat.maxHealth, ImVec2(200, 20));
    
    // Weapon
    const char* weaponNames[] = {"None", "Sword", "Staff", "Hammer"};
    ImGui::Text("Weapon: %s", weaponNames[(int)playerCombat.currentWeapon]);
    
    // Combo
    if (playerCombat.comboCount > 0) {
        ImGui::Text("COMBO: %dx (%.1fx)", playerCombat.comboCount, playerRhythm.rhythmMultiplier);
    }
    
    // Rhythm
    if (playerRhythm.isOnBeat) {
        ImGui::TextColored(ImVec4(1,1,0,1), "ON BEAT!");
    }
    
    ImGui::End();
}
```

---

## Implementation Priority

### Week 1: Core Gameplay
1. **Day 1-2**: PlayerMovementSystem (dash, double-jump, wall-run)
2. **Day 3-4**: CombatSystem integration (attacks, hit detection)
3. **Day 5**: Input system refactor
4. **Day 6-7**: Enemy spawning + basic AI (patrol, chase)

### Week 2: Combat & Feel
1. **Day 8-9**: Enemy attacks + player damage
2. **Day 10-11**: Weapon switching + inventory
3. **Day 12-13**: RhythmSystem + beat timing
4. **Day 14**: Combat feel tuning (hitstop, screen shake)

### Week 3: Polish
1. **Day 15-16**: ImGui HUD
2. **Day 17-18**: Game states (menu, pause, game over)
3. **Day 19-20**: Particle effects for combat
4. **Day 21**: Performance optimization + testing

---

## Testing Strategy

### Unit Tests
- `PlayerMovementSystemTests`: Dash cooldown, double-jump limit
- `CombatSystemTests`: Combo chains, damage calculation
- `RhythmSystemTests`: Beat timing windows, multiplier accuracy
- `EnemyAISystemTests`: State transitions, detection logic

### Integration Tests
- Player attacks enemy, enemy loses health
- Enemy attacks player, player loses health
- Player dies at 0 health, game over state triggered
- Rhythm multiplier applied to combat damage
- Weapon switch updates attack range and visual

### Playtesting Goals
- Movement feels responsive (< 100ms input lag)
- Combat has weight (hitstop on hit)
- Rhythm timing is clear (visual + audio feedback)
- Enemy AI is challenging but fair
- 60 FPS stable with 10+ enemies

---

## Next Immediate Steps

1. **Move WASD input to PlayerMovementSystem** (currently in main loop)
2. **Add dash mechanic** (SHIFT key, force impulse)
3. **Instantiate CombatSystem** in main()
4. **Add mouse click attack** (Left Mouse → Light Attack)
5. **Spawn 3 enemies** in CreateGameWorld()
6. **Basic chase AI** (move toward player)

Once these work, the game becomes playable for the first time!

---

## Files to Modify

### High Priority
- [x] `src_refactored/Demos/Full3DGame_DX12.cpp` - Main game loop, system initialization
- [ ] `src/Gameplay/PlayerMovementSystem.cpp` - Enhanced movement
- [ ] `src/Combat/CombatSystem.cpp` - Combat integration
- [ ] `src/Gameplay/EnemyAISystem.cpp` - Create new file

### Medium Priority
- [ ] `src/Rhythm/RhythmSystem.cpp` - Create new file
- [ ] `include/Gameplay/PlayerMovementSystem.h` - Add PhysX integration
- [ ] `CMakeLists.txt` - Add new source files

### Low Priority (Polish)
- [ ] `src/UI/HUDSystem.cpp` - Create new file with ImGui
- [ ] `src/GameFeel/GameFeelSystem.cpp` - Screen shake, hitstop
- [ ] `src/Particles/ParticleSystem.cpp` - Combat VFX

---

## Current Status Summary

✅ **Working**:
- PhysX physics (gravity, collision, forces)
- D3D12 deferred rendering (141 FPS)
- Basic WASD + jump player movement
- Camera follow system
- ECS component registration (all gameplay components)

🚧 **In Progress**:
- PlayerMovementSystem integration
- CombatSystem connection to game loop

⏳ **Not Started**:
- Enemy AI
- Weapon switching
- Rhythm system
- UI/HUD
- Game states

**Estimated Completion**: 2-3 weeks for fully playable prototype
