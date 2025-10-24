# CudaGame AAA Project Ruleset (Warp)

Version: 1.0
Owner: Brandon (Project Lead)
Applies to: Entire repository; highest precedence for this project unless superseded by a more specific subdirectory ruleset.

1) Primary Directives (Always-On)
- Deliver working, reproducible builds of Full3DGame and EnhancedGame using CMake on Windows x64 with PhysX 5.6.0 and OpenGL 3.3.
- Validate every code suggestion against actual implementation uses before editing (read headers/impls, search references, confirm symbol names and call sites).
- Never delete files; move old versions to an OLD/ folder next to the original path, then add the new version.
- Prefer minimal, safe, incremental patches; compile, run, and verify after each meaningful change. Capture logs.
- Do not commit or push unless explicitly requested. Prepare patches and PR-ready diffs instead.
- Minimize assistant verbosity; focus on actionable diffs and commands. Use TODO planning for 3+ step tasks.

2) Project Charter & Scope
- Vision: A performant third-person action/arena engine with ECS, PhysX-based character/rigid body physics, modern deferred rendering, optional CUDA interop, and strong debug tooling.
- Platform Target: Windows 10/11 x64 (MSVC), NVIDIA GPUs. Future portability is a goal but not a current requirement.
- Core Pillars: Stability, performance, clarity, observability (debug output and metrics), and build/CI reproducibility.

3) Technical Standards (AAA)
- Language/Std: C++17. Prefer RAII; avoid raw new/delete; use unique_ptr/shared_ptr where appropriate. No exceptions for core loops; return error codes or status types.
- Math/Algebra: GLM for math; avoid reinventing primitives.
- Rendering: OpenGL 3.3 core profile with a deferred pipeline (G-buffer + lighting). Keep API state changes minimal. Maintain a RenderDebugSystem for diagnostics.
- Physics: PhysX 5.6.0 (vendored). Ensure /MD runtime and DLL deployment. Character controller + collision remain deterministic within frame.
- CUDA (optional): CUDA-OpenGL interop behind feature flags; gracefully disable when unavailable.
- Assets: All runtime content under assets/. Use compile definition ASSET_DIR and ensure packaging copies assets next to binaries.
- Error Handling: Fail fast on shader/asset load errors with clear messages including absolute path and cwd.
- Performance Budgets (shipping targets):
  - CPU frame: ≤ 16.6 ms (60 FPS) typical scenes; log 95th/99th percentiles.
  - GPU frame: ≤ 16.6 ms typical scenes.
  - Draw calls: target ≤ 1500/frame; warn > 2000.
  - Triangles: target ≤ 8–10M/frame on 3070 Ti equivalent; warn above.
  - Texture binds: target ≤ 500/frame; warn above.
  - Memory: OOM guardrails and leak checks in debug.

4) Build & CI Rules
- CMake Configuration:
  - Always pass -DPHYSX_ROOT_DIR to the vendored PhysX 5.6.0 SDK path.
  - Use /MD runtime across all targets; add /NODEFAULTLIB:LIBCMT /LIBCMTD to avoid runtime conflicts.
  - Define ASSET_DIR for all run targets; keep assets packaged alongside binaries.
- PhysX DLL Handling:
  - Autodetect MSVC bin subdir (e.g., vc142.md/vc143.md) and copy matching DLLs post-build.
  - In CI, parse PHYSX_ROOT_DIR from CMakeCache.txt to locate and copy DLLs.
- CI Pipelines:
  - Pass -DPHYSX_ROOT_DIR and -DENABLE_PHYSX=ON (and -DENABLE_CUDA=ON if present).
  - Separate jobs: (a) Build+package game artifacts; (b) Tests (may be allowed to fail until aligned with refactored API).
  - Upload artifacts with binaries and assets/; include runtime DLLs.
  - Disable pagers for any VCS commands (e.g., git --no-pager).

5) Testing & Quality Gates
- Short-term: Do not block shipping artifacts on legacy test failures; gate tests behind headless GL availability or mock interfaces.
- Medium-term: Align tests with current RenderSystem and ECS APIs; provide minimal TestFramework with ASSERT_* macros and a TestRunner.
- Static Analysis/Style: Prefer clang-format/clang-tidy if configured; otherwise keep style consistent with existing files.
- Verification: After patches, run build, run executable(s), and scan logs for GL/PhysX errors and performance warnings.

6) Source Control & Change Management
- Branching: feature/<name>, bugfix/<name>, hotfix/<name>, release/<tag>.
- Large refactors: stage changes, keep adapters where feasible, migrate tests incrementally. Preserve prior versions under OLD/.
- PR Content: concise description, rationale, benchmarks (if perf-affecting), reproduction steps, logs, and screenshots when relevant.

7) Debugging, Telemetry, and Observability
- Keep OpenGL debug output enabled in debug builds when supported.
- Maintain RenderDebugSystem overlays for G-buffer, depth, wireframe, and stats; log performance warnings.
- Provide targeted dumps (e.g., framebuffer to PPM) only on-demand or when errors occur.

8) Safety, Security, and Secrets
- Never print or log secrets. In commands, reference secrets by env var placeholders (e.g., {{FOO_API_KEY}}). Do not echo secret values.
- Avoid risky or destructive shell commands unless explicitly requested and clearly explained.

9) Assistant Operating Procedure (SOP)
- Intake: Clarify the ask; restate scope if ambiguous.
- Plan: Create TODOs for tasks with ≥3 steps; select lowest-risk incremental path.
- Discover: search_codebase/grep/read files to confirm actual APIs and call sites.
- Implement: apply minimal diffs; follow existing patterns; respect file indentation and style.
- Verify: build with CMake, run target, check logs for errors/warnings; measure impact.
- Document: update notes under UpdatedGameEngineNotes*/ and propose next steps.
- Respect user edits: if the user modifies a suggested diff/command, treat their changes as source of truth.

10) Non-Negotiable Rules (Do/Don’t)
- DO: Check suggestions against actual implementation uses.
- DO: Move old files to OLD/ instead of deleting.
- DO: Keep patches small, reversible, and verified.
- DON’T: Commit or push without explicit instruction.
- DON’T: Introduce new frameworks or major dependencies without approval.
- DON’T: Apply mass reformatting across the repo.
- DON’T: Assume tests or scripts; discover them from the repo and ask if unclear.

11) Definition of Done (per change)
- Builds locally with CMake (Release) and runs the main executable(s) without critical errors.
- PhysX DLLs present next to binaries; shaders/assets resolve via ASSET_DIR.
- Logs show no GL errors; debug warnings triaged or mitigated.
- Notes updated (what changed, why, how to revert) and next steps proposed.

12) Rule Precedence & Evolution
- Subdirectory rules may override this root ruleset for specialized modules; last rule wins on conflict.
- Revisit and revise this ruleset as the engine, CI, and tooling evolve.
