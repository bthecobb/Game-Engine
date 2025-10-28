# Test Automation & Debugging Roadmap

## Current Status (Phase 4.2)

### Test Results Summary
- **Total:** 24 tests
- **Passed:** 14 tests (58.33%) ✅
- **Failed:** 10 tests (41.67%) ❌

### Test Breakdown by Suite

#### ✅ Core Systems (9/9 passed - 100%)
- Entity Creation/Destruction
- Component Add/Remove/Access
- Mass operations (10k entities)
- Performance benchmarks

**Status:** Production-ready

#### ⚠️ Orbit Camera (3/7 passed - 43%)
**Passing:**
- Camera Initialization
- Camera Mode Transitions
- Orbit Settings

**Failing:**
- Camera Movement (distance calculation)
- Camera Zoom (distance not updating)
- Mouse Input (rotation not applied)
- View Projection Matrix (matrix calculations)

**Root Cause:** Distance update logic and matrix recalculation timing issues

#### ⚠️ Character Controller (2/8 passed - 25%)
**Passing:**
- Character Initialization
- (PhysX system retrieval now works!)

**Failing:**
- Basic Movement, Jumping, Double Jump, Sprinting, Wall Running tests

**Root Cause:** Entity ID persistence across tests due to singleton Coordinator state

---

## Immediate Fixes (Sprint 1 - Days 1-2)

### 1. Fix Entity ID Persistence in Tests
**Priority:** CRITICAL
**Impact:** Fixes 5 character controller tests

**Solution:**
```cpp
// Option A: Better Cleanup() in Coordinator
void Coordinator::Cleanup() {
    // Destroy all entities properly
    auto* entityMgr = mEntityManager.get();
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (entityMgr->IsEntityAlive(e)) {
            DestroyEntity(e);
        }
    }
    
    // Reset managers
    mSystemManager = std::make_unique<SystemManager>();
    mComponentManager = std::make_unique<ComponentManager>();
    mEntityManager = std::make_unique<EntityManager>();
}
```

### 2. Fix Orbit Camera Distance Updates
**Priority:** HIGH
**Impact:** Fixes 4 camera tests

**Investigation:**
- Check `OrbitCamera::Update()` distance interpolation
- Verify `ApplyZoom()` modifies settings correctly
- Test `ApplyMouseDelta()` actually rotates camera
- Ensure `UpdateMatrices()` called after parameter changes

---

## Debugging Enhancements (Sprint 2 - Days 3-5)

### 1. Test Debugging Utility Class

**File:** `include_refactored/Testing/TestDebugger.h`

```cpp
namespace CudaGame {
namespace Testing {

class TestDebugger {
public:
    // Entity inspection
    static void DumpEntityState(Core::Entity entity, Core::Coordinator& coord);
    static std::string GetEntityComponentList(Core::Entity entity, Core::Coordinator& coord);
    static std::string GetSystemEntityCounts(Core::Coordinator& coord);
    
    // Test lifecycle logging
    static void LogTestStart(const std::string& testName);
    static void LogTestEnd(const std::string& testName, bool passed, float duration);
    
    // Failure reproduction
    static void SaveFailureContext(const std::string& testName, 
                                    const std::string& error,
                                    Core::Coordinator& coord);
    
    // Performance tracking
    static void RecordTestPerformance(const std::string& testName, float duration);
    static std::string GetPerformanceReport();
    
    // Memory tracking
    static void CheckMemoryLeaks();
    static size_t GetCurrentMemoryUsage();
};

}}
```

### 2. Enhanced Assertion Macros

```cpp
// File: include_refactored/Testing/TestAssertions.h

#define ASSERT_EQ_WITH_CONTEXT(expected, actual, context) \
    if ((expected) != (actual)) { \
        std::stringstream ss; \
        ss << "Assertion failed: " << #expected << " != " << #actual \
           << "\n  Expected: " << (expected) \
           << "\n  Actual: " << (actual) \
           << "\n  Context: " << (context); \
        throw std::runtime_error(ss.str()); \
    }

#define ASSERT_COMPONENT_EXISTS(entity, ComponentType) \
    if (!coordinator->HasComponent<ComponentType>(entity)) { \
        TestDebugger::SaveFailureContext(__func__, "Missing component", *coordinator); \
        std::stringstream ss; \
        ss << "Entity " << entity << " missing " << #ComponentType; \
        throw std::runtime_error(ss.str()); \
    }
```

### 3. Test Reporting Enhancements

**JSON Export Format:**
```json
{
  "test_run": {
    "timestamp": "2025-10-16T02:24:51Z",
    "total_tests": 24,
    "passed": 14,
    "failed": 10,
    "duration_ms": 523,
    "suites": [
      {
        "name": "Core Systems",
        "passed": 9,
        "failed": 0,
        "tests": [
          {
            "name": "Entity Creation",
            "passed": true,
            "duration_ms": 15,
            "benchmarks": {
              "entity_creation_10k": "32 microseconds"
            }
          }
        ]
      }
    ],
    "performance_metrics": {
      "entity_creation_rate": "312,500 entities/sec",
      "component_add_rate": "4,098 ops/sec"
    },
    "failures": [
      {
        "suite": "Orbit Camera",
        "test": "Camera Movement",
        "error": "Distance assertion failed",
        "expected": "<15",
        "actual": "15.9424"
      }
    ]
  }
}
```

---

## Test Automation Pipeline (Sprint 3 - Days 6-10)

### 1. CI/CD Integration

**File:** `.github/workflows/test-automation.yml`

```yaml
name: Automated Testing

on:
  push:
    branches: [ main, develop, feature/* ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  test-windows:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Environment
      run: |
        # Install dependencies
        choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System'
    
    - name: Configure CMake
      run: |
        cmake -B build -DPHYSX_ROOT_DIR=${{env.PHYSX_ROOT}} -DENABLE_PHYSX=ON
    
    - name: Build TestRunner
      run: cmake --build build --config Release --target TestRunner
    
    - name: Run Tests
      run: |
        cd build/bin/tests/Release
        ./TestRunner.exe --output-json test_results.json
    
    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: build/bin/tests/Release/test_results.json
    
    - name: Upload Failure Logs
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: failure-logs
        path: build/bin/tests/Release/failures/
    
    - name: Performance Regression Check
      run: |
        python scripts/check_performance_regression.py \
          --current test_results.json \
          --baseline baseline_results.json \
          --threshold 10
```

### 2. Local Test Automation Script

**File:** `scripts/run_all_tests.ps1`

```powershell
# AAA Test Automation Script
param(
    [switch]$Verbose,
    [switch]$FailFast,
    [string]$Suite = "all",
    [string]$OutputDir = "test_results"
)

Write-Host "=== AAA Engine Test Automation ===" -ForegroundColor Cyan

# Build TestRunner
Write-Host "`nBuilding TestRunner..." -ForegroundColor Yellow
cmake --build build-vs --config Release --target TestRunner
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$OutputDir/test_run_$timestamp.log"

$env:PATH += ";$PWD\build-vs\Release"
.\build-vs\bin\tests\Release\TestRunner.exe --suite $Suite --json "$OutputDir/results_$timestamp.json" | Tee-Object -FilePath $logFile

$exitCode = $LASTEXITCODE

# Parse results
if (Test-Path "$OutputDir/results_$timestamp.json") {
    $results = Get-Content "$OutputDir/results_$timestamp.json" | ConvertFrom-Json
    
    Write-Host "`n=== Test Results ===" -ForegroundColor Cyan
    Write-Host "Total: $($results.total_tests)" -ForegroundColor White
    Write-Host "Passed: $($results.passed) ✓" -ForegroundColor Green
    Write-Host "Failed: $($results.failed) ✗" -ForegroundColor Red
    Write-Host "Success Rate: $([math]::Round($results.passed / $results.total_tests * 100, 2))%" -ForegroundColor Yellow
}

# Check for regressions
if (Test-Path "$OutputDir/baseline_results.json") {
    Write-Host "`nChecking for performance regressions..." -ForegroundColor Yellow
    python scripts/check_regression.py `
        --current "$OutputDir/results_$timestamp.json" `
        --baseline "$OutputDir/baseline_results.json"
}

exit $exitCode
```

### 3. Performance Regression Detection

**File:** `scripts/check_regression.py`

```python
import json
import sys
from pathlib import Path

def check_regression(current_file, baseline_file, threshold_percent=10):
    with open(current_file) as f:
        current = json.load(f)
    
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    regressions = []
    
    # Check test durations
    for suite in current['suites']:
        baseline_suite = next((s for s in baseline['suites'] 
                               if s['name'] == suite['name']), None)
        if not baseline_suite:
            continue
        
        for test in suite['tests']:
            baseline_test = next((t for t in baseline_suite['tests'] 
                                  if t['name'] == test['name']), None)
            if not baseline_test:
                continue
            
            current_duration = test['duration_ms']
            baseline_duration = baseline_test['duration_ms']
            
            if baseline_duration > 0:
                percent_change = ((current_duration - baseline_duration) 
                                  / baseline_duration) * 100
                
                if percent_change > threshold_percent:
                    regressions.append({
                        'test': f"{suite['name']} -> {test['name']}",
                        'baseline': baseline_duration,
                        'current': current_duration,
                        'change_percent': percent_change
                    })
    
    if regressions:
        print("⚠️  PERFORMANCE REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  {r['test']}: {r['baseline']}ms -> {r['current']}ms "
                  f"({r['change_percent']:+.1f}%)")
        return 1
    
    print("✓ No performance regressions detected")
    return 0

if __name__ == "__main__":
    sys.exit(check_regression(sys.argv[1], sys.argv[2]))
```

---

## Code Coverage Analysis (Sprint 4 - Days 11-14)

### Tools Integration
- **OpenCppCoverage** for Windows
- **lcov/gcov** for cross-platform
- **Codecov** for CI integration

### Coverage Targets
- **Core ECS:** 95% (currently ~100%)
- **Rendering:** 70% (needs headless support)
- **Physics:** 80% (good PhysX integration)
- **Gameplay:** 60% (many systems need isolated tests)

**Command:**
```powershell
OpenCppCoverage.exe `
    --sources "C:\Users\Brandon\CudaGame\src_refactored" `
    --sources "C:\Users\Brandon\CudaGame\include_refactored" `
    --excluded_sources "C:\Users\Brandon\CudaGame\external" `
    --export_type html:coverage_report `
    -- .\build-vs\bin\tests\Release\TestRunner.exe
```

---

## Continuous Monitoring (Ongoing)

### Dashboard Metrics
1. **Test Health**
   - Pass rate trend (daily)
   - Flaky test detection
   - Mean time to fix failures

2. **Performance**
   - Entity creation rate
   - Component operation throughput
   - System update times

3. **Code Quality**
   - Test coverage percentage
   - Untested code paths
   - Test-to-code ratio

### Alerts
- **Critical:** Any Core Systems test failure
- **High:** >20% test failure rate
- **Medium:** Performance regression >15%
- **Low:** Coverage drop >5%

---

## Success Criteria

### Phase 4 Complete When:
- ✅ 100% Core Systems passing (DONE)
- ⬜ 100% Camera tests passing
- ⬜ 85%+ Character Controller tests passing
- ⬜ CI pipeline operational
- ⬜ Test coverage >75%
- ⬜ Performance baselines established
- ⬜ Automated regression detection working

### Phase 5 Goals:
- Integration tests for complete game workflows
- Stress testing (100k+ entities)
- Multi-threaded test execution
- Test result analytics dashboard
