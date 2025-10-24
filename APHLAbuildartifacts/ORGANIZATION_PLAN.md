# CudaGame Project Organization Plan

## Purpose
This document defines which files/folders are essential for the current build system vs. which are old/temporary and can be archived.

## Essential Files & Folders (KEEP)

### Build System
- `CMakeLists.txt` - Main build configuration
- `cmake/` - CMake modules and scripts
- `build-vs/` - **Active Visual Studio build directory**
- `.github/` - CI/CD workflows

### Source Code (Current)
- `include_refactored/` - Current headers (C++)
- `src_refactored/` - Current source files (C++)
- `tests/` - Test suites (TestRunner, unit tests)

### Dependencies
- `vendor/` - Third-party libraries (PhysX, etc.)
- `glm/` - GLM math library (header-only)

### Assets & Data
- `assets/` - Shaders, models, textures, etc.

### Scripts & Tools
- `scripts/` - Build scripts, utilities

### Documentation (Current)
- `UpdatedGameEngineNotes1/` - Active development notes
- `README.md` - Project documentation
- `WARP.md` - Warp AI configuration

### Configuration
- `.vscode/` - VS Code settings
- `.sonarcloud/` - Code quality config

## Archival Candidates (MOVE to APHLAbuildartifacts/)

### Old Source Code
- `include/` - **OLD headers** (replaced by include_refactored)
- `src/` - **OLD source** (replaced by src_refactored)
- `temp_physx/` - Temporary PhysX test folder

### Old Build Directories
- `build/` - Generic old build
- `build_clang/` - Clang build artifacts
- `build_msvc/` - MSVC build artifacts  
- `build-vs-nophysx/` - No-PhysX build variant
- `debug_build/` - Old debug build

### CMake Generated (Regenerable)
- `CMakeFiles/` - CMake cache (root level)
- `_deps/` - CMake FetchContent cache
- `*.dir/` folders:
  - `CudaPhysicsDemo.dir/`
  - `CudaRenderingDemo.dir/`
  - `EnhancedGame.dir/`
  - `Full3DGame.dir/`
  - `LightingIntegrationDemo.dir/`
  - `TestRunner.dir/`

### Old Build Outputs
- `Debug/` - Old debug binaries
- `Release/` - Old release binaries
- `x64/` - Old x64 platform outputs
- `bin/` - Old binary output
- `lib/` - Old library output

### Old/Duplicate Content
- `animation/` - Check if duplicates exist in src_refactored
- `external/` - Check if duplicates vendor/
- `docs/` - Check if superseded by UpdatedGameEngineNotes1

### Archives & Captures
- `review/` - Old code reviews
- `screenshots/` - Screenshot captures
- `renderdoccaptures/` - RenderDoc debug captures

### Planning Documents (Optional Archive)
- `AAA_Development_Pipeline/` - High-level planning (keep or archive?)

## TestRunner Specifics

### What TestRunner Needs (KEEP):
```
tests/
├── TestRunner.cpp          # Main test runner
├── CoreSystemsTests.cpp    # ECS tests
├── OrbitCameraTests.cpp    # Camera tests
├── CharacterControllerTests.cpp  # Physics tests
├── PhysXIntegrationTests.cpp     # PhysX tests
└── TestFramework.h         # (actually in include_refactored/Testing/)

include_refactored/Testing/
├── TestFramework.h
├── AdvancedTestFramework.h
└── GPUMetricsStream.h

src_refactored/Testing/
├── TestFramework.cpp
└── AdvancedTestFramework.cpp
```

### What TestRunner Doesn't Need (Can Archive):
- RenderingSystemTests.cpp (currently disabled, needs GL context)
- Full3DGameIntegrationTests.cpp (currently disabled, needs refactoring)
- PlayerMovementTests.cpp (currently disabled, needs GMock)

## Archive Directory Structure

```
APHLAbuildartifacts/
├── old_source/
│   ├── include/         # Old headers
│   ├── src/             # Old source
│   └── temp_physx/      # Temp PhysX folder
├── old_builds/
│   ├── build/
│   ├── build_clang/
│   ├── build_msvc/
│   ├── build-vs-nophysx/
│   └── debug_build/
├── old_outputs/
│   ├── Debug/
│   ├── Release/
│   ├── x64/
│   ├── bin/
│   └── lib/
├── cmake_artifacts/
│   ├── CMakeFiles/
│   ├── _deps/
│   └── *.dir/
├── captures/
│   ├── review/
│   ├── screenshots/
│   └── renderdoccaptures/
└── ORGANIZATION_PLAN.md  # This file
```

## Migration Steps

1. ✅ Create APHLAbuildartifacts/ folder
2. Create subdirectories in APHLAbuildartifacts/
3. Move old source code folders
4. Move old build directories
5. Move old output directories
6. Move CMake artifacts
7. Move captures and archives
8. Verify build still works after cleanup
9. Update .gitignore if needed

## Verification After Cleanup

After moving files, verify:
```bash
# Clean and rebuild
cmake --build build-vs --clean-first --target Full3DGame --config Release
cmake --build build-vs --target TestRunner --config Release

# Test executables exist
Test-Path build-vs/bin/games/Release/Full3DGame.exe
Test-Path build-vs/bin/tests/Release/TestRunner.exe
```

## Safety Notes

- ⚠️ DO NOT DELETE - only move to archive
- ⚠️ Keep vendor/ and glm/ intact (dependencies)
- ⚠️ Keep build-vs/ intact (active build)
- ⚠️ Verify git status before committing
- ⚠️ Test build after each major move

## Current Essential Directory Tree

```
CudaGame/
├── .github/              # CI/CD
├── assets/               # Game assets
├── cmake/                # CMake modules
├── include_refactored/   # Current headers
├── src_refactored/       # Current source
├── tests/                # Test suites
├── vendor/               # Dependencies
├── glm/                  # GLM library
├── build-vs/             # Active build
├── scripts/              # Build scripts
├── UpdatedGameEngineNotes1/  # Docs
├── CMakeLists.txt        # Build config
└── APHLAbuildartifacts/  # Archive folder
```

---
Generated: 2025-10-16
