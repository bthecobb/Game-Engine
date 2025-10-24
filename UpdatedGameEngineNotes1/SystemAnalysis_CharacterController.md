# CharacterController System Analysis

## System Overview

### Architecture
- **Base Class**: Inherits from `Core::System` (ECS System)
- **Namespace**: `CudaGame::Gameplay`
- **Location**:
  - Header: `include_refactored/Gameplay/CharacterControllerSystem.h`
  - Implementation: `src_refactored/Gameplay/CharacterControllerSystem.cpp`

### Core Responsibilities
1. Camera-relative player movement
2. Ground detection and coyote time
3. Jump mechanics (single, double, wall jump)
4. Jump buffering for responsive input
5. Wall-running detection and physics
6. Dash mechanics with cooldown
7. Sprint/walk speed modulation
8. Physics integration via PhysX

---

## Implementation Details

### Component Dependencies

#### Required Components (Per Entity)
```cpp
- Physics::CharacterControllerComponent  // State tracking
- Physics::RigidbodyComponent           // Physics body
- Physics::ColliderComponent            // Collision shape
- Rendering::TransformComponent         // Position/rotation
- Gameplay::PlayerMovementComponent     // Movement parameters
- Gameplay::PlayerInputComponent        // Input state
```

#### System Dependencies
```cpp
- Physics::PhysXPhysicsSystem*          // Physics updates
- Rendering::OrbitCamera*               // Camera-relative movement
```

### State Management

#### CharacterControllerComponent Key Fields
```cpp
// Grounding
bool isGrounded
float lastGroundedTime         // For coyote time
float jumpBufferTimer          // Buffered jump input

// Jumping
bool isJumping
bool canDoubleJump
int airJumps                   // Current air jump count
int maxAirJumps                // Max allowed (usually 1 for double jump)
float jumpForce

// Wall Running
bool isWallRunning
glm::vec3 wallNormal
float wallRunTimer
float maxWallRunTime
float wallRunSpeed

// Dashing
bool isDashing
float dashTimer
float maxDashTime
float dashCooldown
float lastDashTime
glm::vec3 dashDirection
float dashSpeed

// Momentum Preservation
bool shouldPreserveMomentum
glm::vec3 preservedMomentum
```

#### PlayerMovementComponent
```cpp
float baseSpeed = 10.0f
float maxSpeed = 20.0f
float jumpForce = 15.0f
float acceleration
float airAcceleration
float deceleration
```

---

## Update Pipeline

### Main Update Loop
```cpp
void CharacterControllerSystem::Update(float deltaTime) {
    for (auto entity : mEntities) {
        1. UpdateTimers()              // Coyote, jump buffer, dash cooldowns
        2. CheckGrounding()            // Ground detection + landing events
        3. GetCameraRelativeMovement() // Convert input to world-space direction
        4. HandleJump()                // Process jump input with buffering
        5. ApplyMovement()             // Apply forces based on mode
        6. CheckWallRunning()          // Detect and enter/exit wall run
        7. HandleDashing()             // Dash input and state
    }
}
```

### Ground Detection Logic
**Method**: `CheckGrounding()`

**Current Implementation**:
```cpp
const float GROUND_CHECK_DISTANCE = 0.2f;
const float CHARACTER_HEIGHT = 1.8f;

float groundY = 0.0f;  // Hardcoded ground plane
float feetY = transform.position.y - CHARACTER_HEIGHT * 0.5f;

controller.isGrounded = (feetY <= groundY + GROUND_CHECK_DISTANCE) && 
                        (rigidbody.velocity.y <= 0.1f);
```

**Issues**:
- ⚠️ **Hardcoded ground plane** at Y=0 (no actual raycast to PhysX)
- ⚠️ **No multi-surface support** (stairs, slopes, platforms)
- ⚠️ Velocity check `<= 0.1f` allows slight upward motion to count as grounded

---

### Jump System

#### Jump Types
1. **Ground Jump**: Standard jump from ground or within coyote time
2. **Air Jump** (Double Jump): Performed mid-air if enabled
3. **Wall Jump**: Jump off wall surface during wall run

#### Jump Buffering
```cpp
// Jump input registered but can't execute yet? Buffer it!
if (jumpPressedThisFrame) {
    controller.jumpBufferTimer = m_jumpBufferTime;  // 0.1s default
}

// On landing, check buffer
if (controller.jumpBufferTimer > 0) {
    PerformJump();
}
```

**Purpose**: Allows jump input slightly before landing for responsive feel

#### Coyote Time
```cpp
bool canJump = controller.isGrounded || 
               (controller.lastGroundedTime < m_coyoteTime);  // 0.15s default
```

**Purpose**: Allows jump shortly after leaving platform edge

---

### Wall Running System

#### Detection Logic (Lines 340-404)
**Method**: `CheckWallRunning()`

**Current Implementation**:
```cpp
// Only active if:
// 1. Pressing wall run key (GLFW_KEY_E)
// 2. Not grounded
// 3. Not already wall running

// Check 4 directions: Right, Left, Forward, Back
glm::vec3 checkDirections[] = {
    glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0),
    glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)
};

// CURRENT: Checks against world bounds as "walls"
bool nearWall = (transform.position.x > 19.0f && dir.x > 0) ||
                (transform.position.x < -19.0f && dir.x < 0) ||
                // ... etc
```

**Critical Issues**:
- ⚠️ **No actual PhysX raycast** - placeholder logic only
- ⚠️ **Hardcoded world bounds** (-20 to +20 in X/Z)
- ⚠️ **No wall entity detection** - can't detect actual wall objects
- ⚠️ **No angle check** - doesn't validate wall is vertical

#### Wall Run Physics (Lines 318-338)
```cpp
void ApplyWallRunMovement() {
    // Direction perpendicular to wall
    glm::vec3 wallRunDir = glm::cross(wallNormal, glm::vec3(0, 1, 0));
    
    // Match player's current direction
    if (glm::dot(wallRunDir, rigidbody.velocity) < 0) {
        wallRunDir = -wallRunDir;
    }
    
    // Apply velocity
    rigidbody.velocity = wallRunDir * wallRunSpeed;
    rigidbody.velocity.y = 2.0f;  // Counter gravity
    
    // Stick to wall
    glm::vec3 stickForce = -wallNormal * 500.0f;
    rigidbody.addForce(stickForce);
}
```

---

## Test Failures Analysis

### Test Suite Structure
**Location**: `tests/CharacterControllerTests.cpp`  
**Total Tests**: 8  
**Setup**: Creates singleton Coordinator with PhysX, CharacterController, and WallRunning systems

### Critical Setup Code
```cpp
void SetUp() {
    // Use singleton Coordinator (FIXED from previous issues)
    coordinator = std::shared_ptr<Coordinator>(&Coordinator::GetInstance(), [](Coordinator*){});
    coordinator->Cleanup();
    coordinator->Initialize();
    
    // Register systems + set signatures
    // Create player entity with all components
}
```

---

### Failure Analysis

#### 1. Basic Movement Test (Lines 148-165)
**What it tests**: Forward movement (W key) produces velocity and position change

**Test Logic**:
```cpp
input.keys[GLFW_KEY_W] = true;

for (int i = 0; i < 5; i++) {
    characterSystem->Update(FIXED_TIMESTEP);  // 1/60 = 0.0166s
    physicsSystem->Update(FIXED_TIMESTEP);
}

ASSERT_GT(glm::length(rb.velocity), 0.0f);    // Should have velocity
ASSERT_GT(transform.position.z, 0.0f);        // Should have moved forward
```

**Why it fails**:
1. **PhysX not initialized properly in test environment**
   - `PhysXPhysicsSystem::Initialize()` may fail silently
   - Rigidbody forces not integrated without valid PhysX scene
   
2. **Missing camera reference**
   - `GetCameraRelativeMovement()` returns world-space `(0, 0, 1)` when no camera
   - Movement should still work, but might not match expected direction
   
3. **Entity not in system's mEntities set**
   - Even with correct signature, entity might not be tracked
   - **FIXED**: Tests now manually add entity to system sets (lines 92-93)

**Fix Priority**: CRITICAL  
**Estimated Effort**: 3-4 hours

---

#### 2. Jumping Test (Lines 167-182)
**What it tests**: Space key causes upward position change

**Test Logic**:
```cpp
float initialHeight = transform.position.y;  // 2.0f

input.keys[GLFW_KEY_SPACE] = true;

for (int i = 0; i < 5; i++) {
    characterSystem->Update(FIXED_TIMESTEP);
    physicsSystem->Update(FIXED_TIMESTEP);
}

ASSERT_GT(newHeight, initialHeight);  // Should be higher
```

**Why it fails**:
1. **Same PhysX integration issue** as movement test
   - Jump force applied via `rigidbody.addForce()` but not integrated
   
2. **Jump buffer timing**
   - Jump registered on frame 1
   - Grounding check happens before jump logic
   - Might not execute jump if ground state incorrect
   
3. **Static jump state tracking**
   - `static bool jumpPressed` in `HandleJump()` persists across tests
   - Previous test's input state pollutes next test

**Fix Priority**: HIGH  
**Estimated Effort**: 2-3 hours

---

#### 3. Double Jump Test (Lines 184-208)
**What it tests**: Two jumps achieve greater height than one

**Test Logic**:
```cpp
// First jump
input.keys[GLFW_KEY_SPACE] = true;
characterSystem->Update(FIXED_TIMESTEP);

// Release space
input.keys[GLFW_KEY_SPACE] = false;
characterSystem->Update(FIXED_TIMESTEP);

// Second jump
input.keys[GLFW_KEY_SPACE] = true;
for (int i = 0; i < 5; i++) {
    characterSystem->Update(FIXED_TIMESTEP);
}

ASSERT_GT(finalHeight, initialHeight + 2.0f);  // Significant height gain
```

**Why it fails**:
1. **First jump doesn't work** (see Jumping Test failure)
2. **Air jump conditions not met**:
   ```cpp
   if (controller.canDoubleJump && controller.airJumps < controller.maxAirJumps)
   ```
   - `canDoubleJump` defaults to `false` in CharacterControllerComponent
   - Test doesn't enable double jump capability
   
3. **Ground state incorrect**
   - If grounding logic says still on ground, will do two ground jumps
   - Not testing actual double jump mechanic

**Fix Priority**: HIGH  
**Estimated Effort**: 2 hours

---

#### 4. Sprinting Test (Lines 210-232)
**What it tests**: Shift key increases movement speed

**Test Logic**:
```cpp
// Normal movement
input.keys[GLFW_KEY_W] = true;
characterSystem->Update(FIXED_TIMESTEP);
float normalSpeed = glm::length(rb.velocity);

// Sprint
input.keys[GLFW_KEY_LEFT_SHIFT] = true;
for (int i = 0; i < 5; i++) {
    characterSystem->Update(FIXED_TIMESTEP);
}
float sprintSpeed = glm::length(rb.velocity);

ASSERT_GT(sprintSpeed, normalSpeed);           // Faster than normal
ASSERT_LE(sprintSpeed, movement.maxSpeed);     // Capped at max
```

**Why it fails**:
- **Same root cause**: PhysX force integration not working
- **Test design issue**: Compares speeds after different update counts
  - Normal: 1 frame
  - Sprint: 5 frames
  - Should compare steady-state velocities

**Fix Priority**: MEDIUM  
**Estimated Effort**: 1 hour

---

#### 5. Wall Running Detection Test (Lines 235-287)
**What it tests**: Player near wall + E key enables wall running

**Test Logic**:
```cpp
// Create wall entity with WallComponent
Entity wall = coordinator->CreateEntity();
// ... add transform, collider, WallComponent

// Move player next to wall
playerTransform.position = glm::vec3(1.5f, 5.0f, 0.0f);

// Trigger wall run
input.keys[GLFW_KEY_E] = true;
input.keys[GLFW_KEY_W] = true;

for (int i = 0; i < 10; i++) {
    wallRunSystem->Update(FIXED_TIMESTEP);
    characterSystem->Update(FIXED_TIMESTEP);
}

ASSERT_TRUE(controller.isWallRunning);  // Should be wall running
```

**Why it fails**:
1. **Wall detection doesn't check WallComponent**
   - Current code checks position against hardcoded bounds
   - Doesn't query ECS for entities with `WallComponent`
   
2. **No collision query**
   - Needs PhysX overlap/raycast to detect wall collider
   - Test creates proper wall entity but system ignores it
   
3. **Position-based check is wrong**
   - Player at `x=1.5` is NOT near world bound at `x=19.0`
   - Wall at `x=2.0` is not detected by current logic

**Fix Priority**: CRITICAL  
**Estimated Effort**: 4-6 hours (requires PhysX integration)

---

#### 6-8. Wall Running Gravity & Jump Tests
**Depend on**: Wall Running Detection working first

All three tests call `TestWallRunningDetection()` as setup, which fails, causing cascading failures.

---

## System Dependencies & Interactions

### ECS Integration
```
CharacterControllerSystem
├── Requires: PhysX::PhysicsSystem (for force integration)
├── Requires: OrbitCamera (for camera-relative input)
├── Operates on: Entities with signature
│   ├── CharacterControllerComponent
│   ├── RigidbodyComponent
│   ├── ColliderComponent (implicit)
│   ├── TransformComponent
│   ├── PlayerMovementComponent
│   └── PlayerInputComponent
└── Interacts with: WallRunningSystem (separate system)
```

### PhysX Integration Points
1. **Grounding**: Needs raycast down from player feet
2. **Wall Detection**: Needs sphere/box overlap or raycasts in 4 directions
3. **Force Application**: `rigidbody.addForce()` must be integrated by PhysX
4. **Collision Filtering**: Walls need special collision layer
5. **Character Controller**: Should use PhysX CCT (Character Controller) instead of rigidbody

### Input Flow
```
GLFW Input → PlayerInputComponent.keys[] → CharacterControllerSystem
                                           ├→ GetCameraRelativeMovement()
                                           ├→ HandleJump()
                                           ├→ CheckWallRunning()
                                           └→ HandleDashing()
```

---

## Current State Summary

### Working Features ✓
- System initialization and registration
- Component tracking via ECS
- Input reading from PlayerInputComponent
- Camera-relative movement calculation (when camera present)
- State machine logic (grounding, jumping, wall running, dashing)
- Timer management (coyote, jump buffer, cooldowns)
- Momentum preservation on wall exit

### Broken/Incomplete Features ✗
1. **PhysX Integration**
   - No actual physics force application in test environment
   - Ground detection uses hardcoded Y=0 plane
   - No raycasts or collision queries
   
2. **Wall Running System**
   - Hardcoded world bounds instead of detecting WallComponents
   - No PhysX raycast implementation
   - Can't detect actual wall entities
   
3. **Test Isolation**
   - Static variables in HandleJump pollute test state
   - No reset of singleton Coordinator state between tests
   
4. **Double Jump**
   - `canDoubleJump` not set by default
   - Requires explicit enablement (not documented)

### Architecture Issues
1. **Static state** in system methods (should be per-component)
2. **Hardcoded constants** (ground Y, world bounds, character height)
3. **Missing PhysX CCT** (using rigidbody instead of character controller)
4. **No collision layers** (can't filter wall vs floor vs obstacles)

---

## Recommendations

### Critical Fixes (Tests Must Pass)
1. **Initialize PhysX properly in tests**
   - Ensure PxScene is created
   - Verify force integration works
   - Add rigidbody actors to scene
   
2. **Implement wall detection**
   - Query ECS for entities with WallComponent in range
   - Use PhysX overlap sphere/box cast
   - Remove hardcoded world bounds
   
3. **Fix test isolation**
   - Remove static variables from HandleJump/HandleDashing
   - Move state to CharacterControllerComponent
   - Properly reset Coordinator between tests
   
4. **Enable double jump in tests**
   - Set `controller.canDoubleJump = true` in player setup
   - Document requirement in component

### Code Quality Improvements
1. **Replace hardcoded ground check**
   - Implement PhysX raycast from feet downward
   - Support multiple ground surfaces
   - Detect slope angles
   
2. **Refactor wall running**
   - Separate wall detection from wall run logic
   - Support dynamic walls (moving platforms)
   - Add wall angle validation (must be vertical-ish)
   
3. **Use PhysX Character Controller**
   - Replace rigidbody with proper CCT
   - Automatic step climbing
   - Better slope handling
   - Built-in grounding

### Architecture Enhancements
1. **Event system** for state changes (OnJump, OnLand, OnWallRunStart)
2. **Configurable collision layers** for walls/floors/ceilings
3. **Animation integration** (current state → animation parameters)
4. **Network synchronization** readiness (deterministic physics)

---

## Integration with OrbitCamera

### Current Integration
```cpp
void SetCamera(Rendering::OrbitCamera* camera) {
    m_camera = camera;
}
```

### Usage in Movement
```cpp
glm::vec3 GetCameraRelativeMovement() {
    if (!m_camera) {
        return glm::vec3(inputDir.x, 0, inputDir.y);  // World-space fallback
    }
    
    glm::vec3 camForward = m_camera->GetForward();
    camForward.y = 0;  // Project to XZ plane
    camForward = glm::normalize(camForward);
    
    // ... compute camera-relative direction
}
```

### Issues
- Camera pointer set manually (not via dependency injection)
- No null check in all code paths
- Camera update timing not synchronized (can cause jitter)
- Tests don't set camera, using fallback world-space movement

### Recommended Flow
```
1. Input System captures GLFW input
2. OrbitCamera updates view based on mouse
3. CharacterController reads camera orientation
4. CharacterController applies movement
5. PhysX updates positions
6. OrbitCamera follows updated player position
```

**Critical**: Camera must update AFTER CharacterController to avoid frame delay
