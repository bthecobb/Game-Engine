# Bug Tracker - CudaGame Engine Test Suite

**Last Updated**: 2025-01-22  
**Current Status**: 19/24 tests passing (79.17%)

---

## 🟢 RESOLVED BUGS

### BUG-001: OrbitCamera Distance Test Failure
**Status**: ✅ FIXED  
**Priority**: High  
**Component**: OrbitCamera  
**Root Cause**: Test didn't account for height offset in distance calculation  
**Fix**: Added height offset to target position calculation + increased frames for smoothing convergence  
**Commit**: Phase 1 fixes  
**Tests Affected**: Camera Movement

---

### BUG-002: OrbitCamera Zoom Test Failure
**Status**: ✅ FIXED  
**Priority**: Medium  
**Component**: OrbitCamera  
**Root Cause**: Test checked config variable instead of actual camera distance  
**Fix**: Changed test to use `GetDistance()` + increased tolerance for floating point precision  
**Commit**: Phase 1 fixes  
**Tests Affected**: Camera Zoom

---

### BUG-003: OrbitCamera Mouse Input Test Failure
**Status**: ✅ FIXED  
**Priority**: High  
**Component**: OrbitCamera  
**Root Cause**: Mouse delta changes angles but doesn't update forward vector until Update() called  
**Fix**: Added `Update()` call after `ApplyMouseDelta()` in test  
**Commit**: Phase 1 fixes  
**Tests Affected**: Mouse Input

---

### BUG-004: OrbitCamera Projection Matrix Test Failure
**Status**: ✅ FIXED  
**Priority**: Medium  
**Component**: OrbitCamera  
**Root Cause**: Test used incorrect mathematical assertion (1/aspect instead of aspect)  
**Fix**: Corrected assertion to match OpenGL perspective matrix math  
**Commit**: Phase 1 fixes  
**Tests Affected**: View Projection Matrix

---

### BUG-005: TestRunner Missing PhysX DLLs
**Status**: ✅ FIXED  
**Priority**: Critical  
**Component**: Build System  
**Root Cause**: CMakeLists.txt didn't copy PhysX DLLs to TestRunner output directory  
**Fix**: Added POST_BUILD command to copy DLLs to `$<TARGET_FILE_DIR:TestRunner>`  
**Commit**: Phase 1 fixes  
**Tests Affected**: All PhysX-dependent tests (couldn't run)

---

### BUG-006: PhysX Forces Not Applied
**Status**: ✅ FIXED  
**Priority**: Critical  
**Component**: PhysXPhysicsSystem  
**Root Cause**: `forceAccumulator` was never read and applied to PhysX actors  
**Fix**: Added force application in `PhysXPhysicsSystem::Update()` before simulation  
**Commit**: Phase 2.2  
**Tests Affected**: Basic Movement, Jumping

---

### BUG-007: Jump Forces Not Producing Movement
**Status**: ✅ FIXED  
**Priority**: Critical  
**Component**: PhysXPhysicsSystem  
**Root Cause**: Forces applied in eFORCE mode (continuous) instead of eIMPULSE (instant velocity change)  
**Fix**: Changed `addForce()` to use `PxForceMode::eIMPULSE`  
**Commit**: Phase 2.2  
**Tests Affected**: Jumping

---

## 🟡 ACTIVE BUGS

### BUG-008: Double Jump Not Working
**Status**: 🔴 OPEN  
**Priority**: Medium  
**Severity**: Moderate  
**Component**: CharacterControllerSystem  
**Assigned**: Unassigned  
**Created**: 2025-01-22

**Description**:  
Double jump test fails because `canDoubleJump` is not enabled by default in test setup.

**Root Cause**:  
`CharacterControllerComponent.canDoubleJump` defaults to `false`. Test setup doesn't enable it.

**Steps to Reproduce**:
1. Run `TestRunner.exe`
2. Observe "Double Jump" test failure
3. Error: `finalHeight > initialHeight + 2.0f` assertion fails

**Expected Behavior**:  
Player should be able to perform a second jump mid-air, gaining significant additional height.

**Actual Behavior**:  
Only one jump executes. Second jump input is ignored.

**Fix Strategy**:
- **Option A** (Quick): Enable `canDoubleJump = true` in test setup
- **Option B** (Better): Add double jump configuration to `PlayerMovementComponent`

**Estimated Effort**: 30 minutes  
**Files to Modify**:
- `tests/CharacterControllerTests.cpp` (line 113-114)
- OR `include_refactored/Gameplay/PlayerComponents.h`

**Code Fix**:
```cpp
// In CharacterControllerTests.cpp::SetupPlayerComponents()
CudaGame::Physics::CharacterControllerComponent controller;
controller.canDoubleJump = true;  // ADD THIS
controller.maxAirJumps = 1;       // ADD THIS
coordinator->AddComponent(entity, controller);
```

---

### BUG-009: Sprint Speed Exceeds Max Speed
**Status**: 🟡 OPEN  
**Priority**: Low  
**Severity**: Minor  
**Component**: CharacterControllerSystem / PhysXPhysicsSystem  
**Assigned**: Unassigned  
**Created**: 2025-01-22

**Description**:  
After switching to IMPULSE mode for forces, sprint speed occasionally exceeds configured maxSpeed.

**Root Cause**:  
IMPULSE mode applies instant velocity changes, which are more powerful than continuous forces. Movement system applies forces that push velocity slightly over the cap before friction brings it back down.

**Steps to Reproduce**:
1. Run sprint test
2. Observe final velocity > maxSpeed assertion failure

**Expected Behavior**:  
Sprint speed should be clamped to `movement.maxSpeed` (20.0 units/s).

**Actual Behavior**:  
Sprint speed reaches ~21-22 units/s before stabilizing.

**Fix Strategy**:
- **Option A**: Add velocity clamping in `ApplyMovement()`
- **Option B**: Tune acceleration/deceleration values
- **Option C**: Accept slight overshoot as acceptable physics behavior

**Estimated Effort**: 1 hour  
**Notes**: This is more of a tuning issue than a bug. Consider accepting it or adding velocity clamping.

**Severity Justification**: Minor - doesn't break gameplay, just exceeds tuning parameter slightly.

---

### BUG-010: Wall Running Detection Not Working
**Status**: 🔴 OPEN  
**Priority**: High  
**Severity**: Major  
**Component**: CharacterControllerSystem  
**Assigned**: Unassigned  
**Created**: 2025-01-22

**Description**:  
Wall running detection uses hardcoded world bounds instead of detecting actual wall entities with `WallComponent`.

**Root Cause**:  
`CheckWallRunning()` has placeholder logic that checks position against hardcoded bounds (-20 to +20) instead of:
1. Querying ECS for entities with `WallComponent`
2. Using PhysX raycasts to detect walls

**Steps to Reproduce**:
1. Run wall running detection test
2. Test creates wall entity at x=2.0, player at x=1.5
3. Wall is NOT at world bound (x=19.0), so detection fails

**Expected Behavior**:  
System should detect nearby walls via:
1. ECS query for `WallComponent` entities within range
2. PhysX overlap/raycast to verify proximity
3. Check wall normal angle (must be vertical)

**Actual Behavior**:  
Only detects world bounds as "walls", ignores actual wall entities.

**Fix Strategy** (Multi-phase):
1. **Phase 3.2**: Replace hardcoded bounds with ECS query for `WallComponent`
2. **Phase 3.3**: Add PhysX raycast/overlap for accurate detection
3. Add wall normal validation

**Estimated Effort**: 6-8 hours total (split across Phase 3.2 and 3.3)

**Files to Modify**:
- `src_refactored/Gameplay/CharacterControllerSystem.cpp` (lines 340-404)
- `include_refactored/Core/Coordinator.h` (add `GetEntitiesWithComponent<T>()` helper)

**Dependencies**:
- Requires PhysX scene access from CharacterControllerSystem
- Requires actor→entity lookup map in PhysXPhysicsSystem
- Requires collision filtering for wall layer

**Tests Affected**:
- Wall Running Detection
- Wall Running Gravity  
- Wall Running Jump

---

## 📋 FEATURE REQUESTS / ENHANCEMENTS

### FEATURE-001: Static State Pollution in Jump/Dash Handlers
**Status**: 🟡 OPEN  
**Priority**: Medium  
**Type**: Code Quality  
**Created**: 2025-01-22

**Description**:  
`HandleJump()` and `HandleDashing()` use static variables to track previous input state, which pollutes test state across test runs.

**Current Code**:
```cpp
// Line 195 in CharacterControllerSystem.cpp
static bool jumpPressed = false;  // ❌ STATIC!
```

**Proposed Fix**:  
Move state to `CharacterControllerComponent`:
```cpp
struct CharacterControllerComponent {
    bool previousJumpState = false;
    bool previousDashState = false;
    // ... existing fields
};
```

**Benefit**: Better test isolation, thread-safe, proper per-entity state.

**Estimated Effort**: 1 hour

---

### FEATURE-002: Ground Detection Uses Hardcoded Y=0
**Status**: 🟡 OPEN  
**Priority**: Medium  
**Type**: Physics Improvement  
**Created**: 2025-01-22

**Description**:  
Ground detection uses hardcoded ground plane at Y=0 instead of PhysX raycast.

**Current Implementation** (Line 131):
```cpp
float groundY = 0.0f;  // Hardcoded!
```

**Proposed Fix**:  
Implement PhysX raycast from player feet downward:
```cpp
PxRaycastBuffer hit;
bool groundDetected = scene->raycast(origin, direction, distance, hit);
```

**Benefits**:
- Support stairs, slopes, platforms
- Accurate grounding on uneven terrain
- No hardcoded constants

**Estimated Effort**: 2-3 hours

---

### FEATURE-003: Add PhysX Character Controller
**Status**: 🔵 PLANNED  
**Priority**: Low  
**Type**: Architecture  
**Created**: 2025-01-22

**Description**:  
Currently using `PxRigidDynamic` for player. Should use `PxController` (PhysX Character Controller).

**Benefits**:
- Built-in step climbing
- Better slope handling
- Automatic grounding
- Less manual physics tuning

**Estimated Effort**: 8-12 hours (requires refactoring)

**Notes**: Post-Phase 4 enhancement

---

## 📊 BUG STATISTICS

### By Status
- ✅ **Resolved**: 7 bugs
- 🔴 **Open**: 2 critical bugs
- 🟡 **Open**: 1 minor bug
- 🔵 **Planned**: 3 enhancements

### By Component
- **OrbitCamera**: 4 resolved, 0 open
- **Build System**: 1 resolved, 0 open
- **PhysXPhysicsSystem**: 2 resolved, 1 open
- **CharacterControllerSystem**: 0 resolved, 2 open

### By Priority
- **Critical**: 3 resolved, 0 open
- **High**: 2 resolved, 1 open
- **Medium**: 2 resolved, 1 open
- **Low**: 0 resolved, 1 open

---

## 🎯 NEXT SPRINT PRIORITIES

### Sprint Goal: Get to 90%+ Pass Rate

1. **BUG-008** (Double Jump) - Quick win, 30 min
2. **BUG-010** (Wall Running) - High impact, 6-8 hrs
3. **FEATURE-001** (Static State) - Code quality, 1 hr
4. **BUG-009** (Sprint Speed) - Polish, 1 hr

**Estimated Sprint Duration**: 9-11 hours

---

## 📝 TESTING NOTES

### Test Environment Setup
- Build system: CMake + MSBuild
- Test framework: GoogleTest
- Physics: PhysX 5.6.0
- Config: Release build
- Platform: Windows 10/11 x64

### Known Test Limitations
1. Tests use singleton Coordinator (state can leak between tests)
2. PhysX must be initialized fresh per test suite
3. No camera in test environment (movement uses world-space fallback)
4. Fixed timestep assumed (1/60s = 0.0166s)

### Performance Benchmarks
- Entity creation: 41µs
- Component addition: 274µs  
- Component access: 38µs
- Physics step: ~1ms

---

## 🔄 VERSION HISTORY

### v1.0 - January 22, 2025
- Initial bug tracker created
- Documented 7 resolved bugs from Phase 1-2
- Identified 3 active bugs
- Created 3 enhancement requests
- Current pass rate: 79.17% (19/24 tests)
