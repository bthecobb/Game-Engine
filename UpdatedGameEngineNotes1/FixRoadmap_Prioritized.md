# Prioritized Fix Roadmap
## CudaGame Engine - OrbitCamera & CharacterController Systems

**Document Version**: 1.0  
**Date**: 2025-01-21  
**Total Estimated Effort**: 18-28 hours

---

## Executive Summary

### System Status
- **OrbitCamera**: 4/7 tests failing (57% pass rate) - Mostly test issues, system functional
- **CharacterController**: 5/8 tests failing (37% pass rate) - Critical PhysX integration missing

### Root Causes
1. **OrbitCamera**: Test assertions don't match implementation behavior (wrong reference points, timing)
2. **CharacterController**: PhysX not properly initialized in test environment, placeholder logic incomplete

### Strategy
- **Phase 1**: Quick wins (OrbitCamera test fixes) - 2-3 hours
- **Phase 2**: CharacterController foundations (PhysX integration) - 8-12 hours  
- **Phase 3**: Advanced features (wall running, double jump) - 6-10 hours
- **Phase 4**: Polish and refactoring - 2-3 hours

---

## Phase 1: OrbitCamera Test Fixes (Quick Wins)
**Total Time**: 2-3 hours  
**Risk**: LOW  
**Priority**: HIGH (establishes working baseline)

### 1.1 Fix Projection Matrix Test ⏱️ 15 min
**File**: `tests/OrbitCameraTests.cpp` (Line 147)  
**Issue**: Wrong mathematical assertion

**Current**:
```cpp
ASSERT_NEAR(projMatrix[1][1] / projMatrix[0][0], 1.0f/aspect, EPSILON);
```

**Fix**:
```cpp
ASSERT_NEAR(projMatrix[1][1] / projMatrix[0][0], aspect, EPSILON);
```

**Verification**: Test passes immediately

---

### 1.2 Fix Zoom Test ⏱️ 30 min
**Files**: 
- `tests/OrbitCameraTests.cpp` (Lines 108-118)
- `include_refactored/Rendering/OrbitCamera.h` (optional getter addition)

**Issue**: Test checks `GetOrbitSettings().distance` (config) instead of actual camera distance

**Option A - Fix Test Only** (15 min):
```cpp
// Change line 109 & 113 & 117
float initialDistance = camera->GetDistance();  // Was: GetOrbitSettings().distance
// ...
ASSERT_LT(camera->GetDistance(), initialDistance);
```

**Option B - Improve API** (30 min):
Add to `OrbitCamera.h`:
```cpp
float GetTargetDistance() const { return m_targetDistance; }
```

Update test to use `GetTargetDistance()` for clearer semantics

**Recommendation**: Option B (better long-term API)

---

### 1.3 Fix Mouse Input Test ⏱️ 30 min
**File**: `tests/OrbitCameraTests.cpp` (Lines 121-131)

**Issue**: Mouse delta changes angles but doesn't update forward vector immediately

**Option A - Fix Test** (10 min):
```cpp
glm::vec3 initialForward = camera->GetForward();
camera->ApplyMouseDelta(10.0f, 5.0f);
camera->Update(0.0f, glm::vec3(0), glm::vec3(0));  // Force update with zero dt
glm::vec3 newForward = camera->GetForward();
```

**Option B - Improve System** (30 min):
Modify `OrbitCamera::ApplyMouseDelta()` to immediately update vectors:
```cpp
void OrbitCamera::ApplyMouseDelta(float xDelta, float yDelta) {
    m_yaw += xDelta * m_orbitSettings.mouseSensitivity;
    m_pitch += yDelta * m_orbitSettings.mouseSensitivity;
    ClampAngles();
    
    // NEW: Immediate update for responsive input
    UpdateCameraVectorsFromPosition();
}
```

**Recommendation**: Option A for now (less risky), Option B for future improvement

---

### 1.4 Fix Camera Movement Test ⏱️ 1-2 hours
**File**: `tests/OrbitCameraTests.cpp` (Lines 91-105)

**Issue 1**: Test compares distance to wrong reference point (doesn't account for height offset)

**Issue 2**: Smoothing delay after only 5 frames

**Fix Strategy**:

**Part 1 - Fix Reference Point** (30 min):
```cpp
// Line 103-104: Account for height offset
glm::vec3 actualTarget = targetPos + glm::vec3(0, camera->GetOrbitSettings().heightOffset, 0);
float distance = glm::distance(cameraPos, actualTarget);
ASSERT_GT(distance, camera->GetOrbitSettings().distance - 1.0f);
ASSERT_LT(distance, camera->GetOrbitSettings().distance + 1.0f);
```

**Part 2 - Fix Smoothing** (30 min):
Option A: Increase frame count (quick):
```cpp
for (int i = 0; i < 30; i++) {  // Was: 5
```

Option B: Add helper method to `OrbitCamera` (better):
```cpp
void OrbitCamera::SetPositionImmediate(const glm::vec3& target) {
    m_targetPosition = target + glm::vec3(0, m_orbitSettings.heightOffset, 0);
    m_desiredPosition = CalculateDesiredPosition(m_targetPosition);
    m_currentPosition = m_desiredPosition;  // No smoothing
    m_currentDistance = m_targetDistance;
    UpdateCameraVectorsFromPosition();
}
```

**Recommendation**: Part 1 + Option A (simple, effective)

---

## Phase 2: CharacterController PhysX Integration
**Total Time**: 8-12 hours  
**Risk**: HIGH  
**Priority**: CRITICAL (blocks all other character tests)

### 2.1 Diagnose PhysX Test Environment ⏱️ 2-3 hours
**Files**: 
- `src_refactored/Physics/PhysXPhysicsSystem.cpp`
- `tests/CharacterControllerTests.cpp`

**Tasks**:
1. Add verbose logging to `PhysXPhysicsSystem::Initialize()`
2. Verify PxScene creation in test environment
3. Check PxFoundation, PxPhysics initialization
4. Verify error handling and silent failures

**Debugging Checklist**:
```cpp
✓ PxCreateFoundation() succeeds
✓ PxCreatePhysics() succeeds
✓ PxScene created with valid descriptor
✓ Gravity set correctly
✓ Update() method processes scene simulation
✓ Force accumulation works
```

**Deliverable**: Diagnostic report on PhysX state in tests

---

### 2.2 Fix PhysX Actor Creation ⏱️ 3-4 hours
**Files**:
- `src_refactored/Physics/PhysXPhysicsSystem.cpp`
- `tests/CharacterControllerTests.cpp` (SetUp method)

**Issue**: RigidbodyComponent forces not integrated because no PxRigidDynamic actor exists

**Implementation**:
1. **Add entity→actor mapping**:
```cpp
// In PhysXPhysicsSystem.h
std::unordered_map<Entity, physx::PxRigidDynamic*> m_actorMap;
```

2. **Create actors in test setup**:
```cpp
// In CharacterControllerTests::SetupPlayerComponents()
// After adding RigidbodyComponent:
physicsSystem->CreateActor(player);  // New method to implement
```

3. **Implement PhysXPhysicsSystem::CreateActor()**:
```cpp
void PhysXPhysicsSystem::CreateActor(Entity entity) {
    auto& rb = coordinator->GetComponent<RigidbodyComponent>(entity);
    auto& transform = coordinator->GetComponent<TransformComponent>(entity);
    auto& collider = coordinator->GetComponent<ColliderComponent>(entity);
    
    // Create PxShape from collider
    PxShape* shape = CreateShape(collider);
    
    // Create dynamic actor
    PxTransform pxTransform(ToPxVec3(transform.position), PxQuat(PxIdentity));
    PxRigidDynamic* actor = PxCreateDynamic(*m_physics, pxTransform, *shape, 10.0f);
    actor->setMass(rb.mass);
    
    // Add to scene
    m_scene->addActor(*actor);
    m_actorMap[entity] = actor;
}
```

4. **Update force application**:
```cpp
void PhysXPhysicsSystem::Update(float dt) {
    // Apply forces from rigidbody components
    for (auto entity : mEntities) {
        if (m_actorMap.count(entity)) {
            auto* actor = m_actorMap[entity];
            auto& rb = coordinator->GetComponent<RigidbodyComponent>(entity);
            
            actor->addForce(ToPxVec3(rb.force));
            rb.force = glm::vec3(0);  // Clear after application
        }
    }
    
    // Step simulation
    m_scene->simulate(dt);
    m_scene->fetchResults(true);
    
    // Read back results
    for (auto entity : mEntities) {
        if (m_actorMap.count(entity)) {
            auto* actor = m_actorMap[entity];
            auto& transform = coordinator->GetComponent<TransformComponent>(entity);
            auto& rb = coordinator->GetComponent<RigidbodyComponent>(entity);
            
            PxTransform pxTransform = actor->getGlobalPose();
            transform.position = ToGlmVec3(pxTransform.p);
            rb.velocity = ToGlmVec3(actor->getLinearVelocity());
        }
    }
}
```

**Verification**: Movement and jump tests should pass

---

### 2.3 Fix Static State Pollution ⏱️ 1 hour
**File**: `src_refactored/Gameplay/CharacterControllerSystem.cpp`

**Issue**: Static variables in `HandleJump()` and `HandleDashing()` persist across tests

**Current** (Line 195-197):
```cpp
void CharacterControllerSystem::HandleJump(...) {
    static bool jumpPressed = false;  // ❌ STATIC!
    bool jumpThisFrame = input.keys[GLFW_KEY_SPACE] && !jumpPressed;
    jumpPressed = input.keys[GLFW_KEY_SPACE];
```

**Fix - Move to Component**:
```cpp
// In Physics/CharacterController.h (CharacterControllerComponent):
bool previousJumpState = false;
bool previousDashState = false;
```

**Updated HandleJump()**:
```cpp
void CharacterControllerSystem::HandleJump(...) {
    bool jumpThisFrame = input.keys[GLFW_KEY_SPACE] && !controller.previousJumpState;
    controller.previousJumpState = input.keys[GLFW_KEY_SPACE];
    // ... rest of logic
}
```

**Files to Update**:
- `include_refactored/Physics/CharacterController.h` (add fields)
- `src_refactored/Gameplay/CharacterControllerSystem.cpp` (lines 195-197, 421-423)

---

### 2.4 Implement Ground Raycasting ⏱️ 2-3 hours
**File**: `src_refactored/Gameplay/CharacterControllerSystem.cpp` (Lines 118-152)

**Current**: Hardcoded Y=0 ground plane  
**Goal**: PhysX raycast from feet downward

**Implementation**:
```cpp
void CharacterControllerSystem::CheckGrounding(...) {
    const float GROUND_CHECK_DISTANCE = 0.2f;
    const float CHARACTER_HEIGHT = 1.8f;
    
    glm::vec3 rayStart = transform.position;
    glm::vec3 rayEnd = rayStart - glm::vec3(0, CHARACTER_HEIGHT * 0.5f + GROUND_CHECK_DISTANCE, 0);
    
    // PhysX raycast
    PxVec3 origin = ToPxVec3(rayStart);
    PxVec3 direction = ToPxVec3(glm::normalize(rayEnd - rayStart));
    float distance = glm::distance(rayStart, rayEnd);
    
    PxRaycastBuffer hit;
    bool groundDetected = m_physicsSystem->GetScene()->raycast(
        origin, direction, distance, hit
    );
    
    bool wasGrounded = controller.isGrounded;
    controller.isGrounded = groundDetected && (rigidbody.velocity.y <= 0.1f);
    
    // Landing detection...
}
```

**Dependencies**: PhysX scene must be accessible from CharacterControllerSystem

---

## Phase 3: Advanced Character Features
**Total Time**: 6-10 hours  
**Risk**: MEDIUM  
**Priority**: MEDIUM (depends on Phase 2 completion)

### 3.1 Enable Double Jump in Tests ⏱️ 30 min
**File**: `tests/CharacterControllerTests.cpp` (Line 134)

**Simple Fix**:
```cpp
void SetupPlayerComponents(Entity entity) {
    // ... existing setup ...
    
    // Enable double jump for testing
    auto& controller = coordinator->GetComponent<CharacterControllerComponent>(entity);
    controller.canDoubleJump = true;
    controller.maxAirJumps = 1;
}
```

**Better Fix** - Add to PlayerMovementComponent:
```cpp
// In Gameplay/PlayerComponents.h
struct PlayerMovementComponent {
    // ... existing fields ...
    bool canDoubleJump = true;
    int maxAirJumps = 1;
};

// CharacterControllerSystem reads from movement component instead
```

---

### 3.2 Implement Wall Detection via ECS Query ⏱️ 3-4 hours
**File**: `src_refactored/Gameplay/CharacterControllerSystem.cpp` (Lines 340-404)

**Goal**: Replace hardcoded world bounds with actual wall entity detection

**Implementation**:
```cpp
void CharacterControllerSystem::CheckWallRunning(...) {
    if (!input.keys[GLFW_KEY_E] || controller.isGrounded) {
        if (controller.isWallRunning) ExitWallRun(controller);
        return;
    }
    
    // Query nearby wall entities
    auto& coordinator = Coordinator::GetInstance();
    std::vector<Entity> nearbyWalls;
    
    // Get all entities with WallComponent
    for (auto wallEntity : coordinator.GetEntitiesWithComponent<WallComponent>()) {
        auto& wallTransform = coordinator.GetComponent<TransformComponent>(wallEntity);
        float distance = glm::distance(transform.position, wallTransform.position);
        
        if (distance < WALL_DETECTION_RANGE) {
            nearbyWalls.push_back(wallEntity);
        }
    }
    
    // For each nearby wall, check direction
    for (auto wall : nearbyWalls) {
        auto& wallTransform = coordinator.GetComponent<TransformComponent>(wall);
        auto& wallCollider = coordinator.GetComponent<ColliderComponent>(wall);
        
        // Calculate wall normal (simplified for box colliders)
        glm::vec3 toWall = wallTransform.position - transform.position;
        glm::vec3 wallNormal = CalculateWallNormal(wallTransform, wallCollider, transform.position);
        
        // Check if moving toward wall
        float approachSpeed = glm::dot(rigidbody.velocity, -wallNormal);
        if (approachSpeed > 2.0f) {
            // Start wall running
            controller.isWallRunning = true;
            controller.wallNormal = wallNormal;
            controller.wallRunTimer = 0;
            break;
        }
    }
}

glm::vec3 CalculateWallNormal(const TransformComponent& wallTransform,
                              const ColliderComponent& wallCollider,
                              const glm::vec3& playerPos) {
    // For box colliders, find closest face
    glm::vec3 toPlayer = playerPos - wallTransform.position;
    
    // Project onto each axis
    float xDist = std::abs(toPlayer.x);
    float yDist = std::abs(toPlayer.y);
    float zDist = std::abs(toPlayer.z);
    
    // Return normal of closest face
    if (xDist > yDist && xDist > zDist) {
        return glm::vec3(glm::sign(toPlayer.x), 0, 0);
    } else if (zDist > yDist) {
        return glm::vec3(0, 0, glm::sign(toPlayer.z));
    }
    return glm::vec3(0, 1, 0);  // Top/bottom (invalid for wall run)
}
```

**Additional Requirements**:
- Add `GetEntitiesWithComponent<T>()` helper to Coordinator (1 hour)
- Define `WALL_DETECTION_RANGE` constant (suggest 1.5f)

---

### 3.3 PhysX Raycast for Wall Detection ⏱️ 2-3 hours
**Goal**: Replace distance check with proper collision detection

**Implementation**:
```cpp
void CharacterControllerSystem::CheckWallRunning(...) {
    // ... E key check ...
    
    const float WALL_CHECK_DISTANCE = 1.5f;
    glm::vec3 checkDirections[] = {
        glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)
    };
    
    for (const auto& dir : checkDirections) {
        PxVec3 origin = ToPxVec3(transform.position);
        PxVec3 direction = ToPxVec3(dir);
        
        PxRaycastBuffer hit;
        bool hitWall = m_physicsSystem->GetScene()->raycast(
            origin, direction, WALL_CHECK_DISTANCE, hit,
            PxHitFlag::eDEFAULT,
            PxQueryFilterData(PxQueryFlag::eSTATIC)  // Only static geometry
        );
        
        if (hitWall) {
            // Check if hit entity has WallComponent
            Entity hitEntity = GetEntityFromActor(hit.block.actor);
            if (coordinator.HasComponent<WallComponent>(hitEntity)) {
                auto& wallComp = coordinator.GetComponent<WallComponent>(hitEntity);
                if (wallComp.canWallRun) {
                    // Start wall run
                    controller.isWallRunning = true;
                    controller.wallNormal = ToGlmVec3(hit.block.normal);
                    break;
                }
            }
        }
    }
}
```

**Dependencies**:
- Actor→Entity lookup map in PhysXPhysicsSystem
- Collision filtering layers

---

## Phase 4: Polish & Refactoring
**Total Time**: 2-3 hours  
**Risk**: LOW  
**Priority**: LOW (quality of life improvements)

### 4.1 Add Integration Tests ⏱️ 1 hour
**File**: `tests/CharacterControllerIntegrationTests.cpp` (new)

**Purpose**: Test full update cycle with camera + movement

```cpp
void TestCameraRelativeMovement() {
    // Setup player + camera
    // Press W key
    // Update: Input → CharacterController → Physics → Camera
    // Verify: Player moved in camera's forward direction
}

void TestWallRunToJump() {
    // Setup player near wall
    // Start wall run
    // Press space
    // Verify: Exit wall run + jump away from wall
}
```

---

### 4.2 Refactor Hardcoded Constants ⏱️ 30 min
**Files**: Multiple

**Goal**: Move magic numbers to configuration

**Create**: `include_refactored/Gameplay/CharacterControllerConfig.h`
```cpp
namespace CudaGame::Gameplay {
    struct CharacterControllerConfig {
        // Grounding
        float groundCheckDistance = 0.2f;
        float characterHeight = 1.8f;
        float groundedVelocityThreshold = 0.1f;
        
        // Wall Running
        float wallCheckDistance = 1.5f;
        float wallRunMinApproachSpeed = 2.0f;
        
        // Physics
        float defaultMass = 80.0f;
        glm::vec3 defaultColliderSize = glm::vec3(0.8f, 1.8f, 0.8f);
    };
}
```

---

### 4.3 Add Diagnostic Logging ⏱️ 30 min
**Files**: `CharacterControllerSystem.cpp`, `OrbitCamera.cpp`

**Goal**: Optional verbose logging for debugging

```cpp
class CharacterControllerSystem {
    bool m_debugLogging = false;
public:
    void SetDebugLogging(bool enable) { m_debugLogging = enable; }
    
    void CheckGrounding(...) {
        // ... logic ...
        if (m_debugLogging && wasGrounded != controller.isGrounded) {
            std::cout << "[CharacterController] Grounding state changed: "
                     << (controller.isGrounded ? "GROUNDED" : "AIRBORNE") << "\n";
        }
    }
};
```

Enable in tests:
```cpp
characterSystem->SetDebugLogging(true);  // Verbose test output
```

---

### 4.4 Documentation Updates ⏱️ 30 min
**Files**: README additions, component documentation

**Tasks**:
1. Document CharacterControllerComponent fields
2. Add usage examples for double jump, wall run
3. Update build/test instructions
4. Create troubleshooting guide

---

## Implementation Order (Recommended)

### Week 1: Foundation (10-15 hours)
```
Day 1-2: Phase 1 (OrbitCamera tests) → 3 hours
         Phase 2.1 (PhysX diagnosis) → 3 hours
Day 3-4: Phase 2.2 (Actor creation) → 4 hours
Day 5:   Phase 2.3 (Static state) → 1 hour
         Phase 2.4 (Ground raycast) → 3 hours
```

**Milestone**: Basic movement and jumping tests pass

### Week 2: Features (8-12 hours)
```
Day 1:   Phase 3.1 (Double jump) → 30 min
         Phase 3.2 (Wall detection) → 4 hours
Day 2-3: Phase 3.3 (Wall raycasts) → 3 hours
Day 4:   Phase 4 (Polish) → 3 hours
```

**Milestone**: All tests pass, production ready

---

## Risk Mitigation

### High-Risk Items
1. **PhysX Integration** (Phase 2.2)
   - **Risk**: PxScene setup failures, actor lifecycle bugs
   - **Mitigation**: Incremental testing, verbose logging, reference PhysX docs
   
2. **Wall Detection** (Phase 3.2-3.3)
   - **Risk**: Complex collision logic, edge cases
   - **Mitigation**: Start with simplified box colliders, add complexity gradually

### Dependencies
```
Phase 1 → Independent (can start immediately)
Phase 2.1 → Blocks all Phase 2
Phase 2.2 → Blocks Phase 2.3, 2.4
Phase 2.4 → Blocks Phase 3
Phase 3.1 → Depends on Phase 2 completion
Phase 3.2 → Depends on Phase 2 completion
Phase 3.3 → Depends on Phase 3.2
Phase 4 → Can parallelize with Phase 3
```

---

## Success Metrics

### Phase 1 Complete
- ✓ OrbitCamera: 7/7 tests passing
- ✓ No regressions in existing functionality

### Phase 2 Complete
- ✓ PhysX scene initializes in tests
- ✓ Movement test passes (velocity > 0)
- ✓ Jump test passes (height increases)
- ✓ Sprint test passes (speed difference detected)

### Phase 3 Complete
- ✓ Double jump test passes
- ✓ Wall running detection test passes
- ✓ Wall running gravity test passes
- ✓ Wall jump test passes

### Phase 4 Complete
- ✓ Integration tests added
- ✓ Configuration externalized
- ✓ Documentation complete

---

## Maintenance Plan

### After Fixes Complete
1. **Add CI test automation** (GitHub Actions, Jenkins, etc.)
2. **Performance profiling** (PhysX overhead, ECS queries)
3. **Code coverage analysis** (aim for >80%)
4. **Refactoring backlog** (PhysX CCT migration, event system)

### Ongoing
- Monitor test stability (flaky tests)
- Update documentation as features added
- Review PhysX best practices periodically
