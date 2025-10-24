# Next Steps and Assessment

## Assessment (Current State)
- Build: Configures and builds in Release with PhysX 5.6.0 via PHYSX_ROOT_DIR; executables run.
- Runtime: OpenGL 3.3 initializes; deferred pipeline works; depth blit error resolved; debug output active.
- Tests: Multiple compilation errors due to API/infra mismatches (RenderSystem interface, test framework, ECS types).
- CI: PhysX paths and flags inconsistent; risk of DLL mismatch; some flags ignored (-DPHYSX_ROOT vs -DPHYSX_ROOT_DIR).
- Assets: ASSET_DIR passed from CMake; ensure assets are present with binaries.

## Immediate Next Steps (Priority)
1) CI Consistency
- Update all workflows to pass -DPHYSX_ROOT_DIR and -DENABLE_PHYSX=ON.
- Parse PHYSX_ROOT_DIR from CMakeCache.txt and copy DLLs from ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release (or detected compiler dir).

2) CMake Hardening
- Add logic to discover PhysX bin subdir dynamically (vc142.md, vc143.md) and set PHYSX_DLL_DIR accordingly.
- Expose PHYSX_DLL_DIR via a cache variable and reuse in post-build copy rules for all targets.

3) Tests Unblock Plan (short-term)
- Temporarily set BUILD_TESTING=OFF in CI for release artifact builds.
- Or split CI jobs: (a) Build+package game, (b) Build tests (allowed to fail) until fixed.

4) Tests Fix Plan (medium-term)
- Align tests with actual RenderSystem API (no InitializeGBuffer/Begin*/End* in current class).
- Provide minimal TestFramework with ASSERT_* macros and a trivial TestRunner.
- Gate GL-dependent tests behind headless context availability (or mock interfaces at a higher level).

5) Packaging
- Add post-build step to copy assets/ to $<TARGET_FILE_DIR:...>/assets for Full3DGame and EnhancedGame.
- Verify shader paths resolve at runtime with relative ASSET_DIR.

6) Logging & Robustness
- Add clearer shader load error messages (path, cwd) and early-exit guards.
- Ensure RenderSystem calls RenderDebugSystem BeginFrame/EndFrame and updates RenderStatistics each frame.

## Optional Cleanup (Low Risk)
- Comment or move unused PhysX folders (e.g., submodule) to OLD/ to reduce confusion (do not delete).

## Stretch Goals
- Implement cel-shading/outline passes in a feature flag path.
- Add shader hot-reload watcher for assets/shaders.

## Definition of Done for this iteration
- CI builds and runs Full3DGame/EnhancedGame reliably with PhysX 5.6.0.
- Assets copied alongside binaries; game runs from CI artifact.
- Tests gated or partially fixed to compile in CI without blocking artifacts.
- CMake autodetects PhysX bin subdir and copies matching DLLs.
