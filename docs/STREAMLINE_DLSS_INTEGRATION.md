# NVIDIA Streamline SDK Integration for DLSS

## Status: In Progress ✅

**Date**: November 18, 2025  
**SDK Version**: Streamline 2.9.0  
**Target**: D3D12 Backend with DLSS Super Resolution

---

## Overview

NVIDIA Streamline SDK is the official integration framework for:
- **DLSS Super Resolution** (upscaling)
- **DLSS Frame Generation** (DLSS 3)
- **NVIDIA Reflex** (low-latency)
- **RTXGI** (ray-traced global illumination)
- **NIS** (NVIDIA Image Scaling)

Streamline provides a unified API across D3D12, D3D11, and Vulkan with plugin architecture.

---

## Why Streamline Instead of Direct DLSS SDK?

### Previous Approach (Abandoned)
- Downloaded `nvngx_dlss_310.4.0` SDK
- **Problem**: Only includes D3D11 support (`nvsdk_ngx_helpers_dlssd.h`)
- No D3D12-specific headers or functions
- Would require low-level NGX integration

### Current Approach (Streamline)
- ✅ Full D3D12 support via `sl_dlss.h`
- ✅ Simplified API - handles NGX internally
- ✅ Hot-swappable DLSS versions (OTA updates)
- ✅ Includes all required DLLs and libraries
- ✅ Production-ready, used by AAA titles

---

## SDK Location

```
vendor/streamline-sdk/
├── bin/x64/
│   ├── sl.interposer.dll       # Streamline core
│   ├── sl.dlss.dll             # DLSS plugin
│   ├── nvngx_dlss.dll          # DLSS runtime
│   ├── sl.reflex.dll           # Reflex plugin
│   └── ...
├── include/
│   ├── sl.h                    # Core API
│   ├── sl_dlss.h               # DLSS API
│   ├── sl_reflex.h             # Reflex API
│   └── ...
├── lib/x64/
│   └── sl.interposer.lib       # Link-time library
└── docs/
    ├── ProgrammingGuideDLSS.md
    └── ...
```

---

## Integration Plan

### Phase 1: Core Streamline Integration ✅
- [x] Download and extract Streamline SDK 2.9.0
- [x] Move to `vendor/streamline-sdk/`
- [x] Create `StreamlineDLSSWrapper.h` header
- [ ] Create `StreamlineDLSSWrapper.cpp` implementation
- [ ] Update CMakeLists.txt for Streamline

### Phase 2: DLSS Initialization
- [ ] Initialize Streamline with `slInit()`
- [ ] Set feature `kFeatureDLSS` enabled
- [ ] Query optimal render resolution
- [ ] Create viewport handle

### Phase 3: Per-Frame Integration
- [ ] Tag D3D12 resources (color, depth, motion vectors)
- [ ] Set DLSS options (quality mode, output resolution)
- [ ] Evaluate DLSS via Streamline
- [ ] Handle jitter offsets for TAA

### Phase 4: Testing
- [ ] Unit tests for Streamline initialization
- [ ] Test quality mode switching
- [ ] Measure performance (FPS with/without DLSS)
- [ ] Verify temporal stability (no ghosting)

### Phase 5: Advanced Features
- [ ] DLSS Frame Generation (DLSS 3)
- [ ] Dynamic resolution scaling
- [ ] Per-viewport DLSS settings
- [ ] Reflex integration

---

## API Usage

### Initialization
```cpp
// 1. Initialize Streamline
sl::Preferences prefs = {};
prefs.showConsole = true;
prefs.logLevel = sl::LogLevel::eDefault;
prefs.flags = sl::PreferenceFlags::eUseDefferedContexts;
slInit(prefs, adapter);

// 2. Set DLSS feature enabled
sl::FeatureRequirements reqs = {};
reqs.flags = sl::FeatureFlags::eD3D12;
slSetFeatureLoaded(sl::kFeatureDLSS, true);

// 3. Query optimal settings
sl::DLSSOptions options = {};
options.mode = sl::DLSSMode::eMaxQuality;
options.outputWidth = 3840;
options.outputHeight = 2160;

sl::DLSSOptimalSettings settings = {};
slDLSSGetOptimalSettings(options, settings);
// Use settings.optimalRenderWidth/Height for rendering
```

### Per-Frame Execution
```cpp
// 1. Tag resources
sl::ResourceTag colorTag = {};
colorTag.type = sl::ResourceType::eTex2d;
colorTag.resource = colorBuffer;
colorTag.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
slSetTag(viewport, &colorTag, sl::ResourceLifecycle::eValidUntilPresent);

// 2. Set DLSS options
slDLSSSetOptions(viewport, options);

// 3. Evaluate (automatic via tagged command list)
// Streamline intercepts Present() and injects DLSS
```

---

## Performance Targets (RTX 3070 Ti @ 4K)

| Mode | Render Res | Upscale | Target FPS |
|------|-----------|---------|------------|
| **UltraPerformance** | 1080p | 3.0x | 120+ FPS |
| **Performance** | 1440p | 2.0x | 90+ FPS |
| **Balanced** | 1662p | 1.7x | 75+ FPS |
| **Quality** | 1800p | 1.5x | 60+ FPS |
| **UltraQuality** | 2880p | 1.3x | 45+ FPS |
| **DLAA** | 4K | 1.0x | 30+ FPS (best AA) |

---

## Required DLLs for Runtime

Must be copied to executable directory:
```
sl.interposer.dll        # Core Streamline
sl.dlss.dll              # DLSS plugin
nvngx_dlss.dll           # DLSS 3.x runtime
sl.common.dll            # Utilities
```

Optional for other features:
```
sl.reflex.dll            # Low-latency
sl.dlss_g.dll            # Frame Generation (RTX 40 series)
sl.nis.dll               # Fallback upscaling
```

---

## Key Differences from Direct NGX

| Feature | Direct NGX SDK | Streamline |
|---------|---------------|------------|
| **API Complexity** | Low-level, manual setup | High-level, automatic |
| **D3D12 Support** | Partial | Full |
| **OTA Updates** | No | Yes (DLL hot-swap) |
| **Multi-Feature** | DLSS only | DLSS + Reflex + RTXGI + more |
| **Maintenance** | Manual NGX calls | Automatic via tagging |
| **Production Use** | Discouraged | Recommended by NVIDIA |

---

## Build Configuration

### CMake Changes Needed
```cmake
# Streamline SDK paths
set(STREAMLINE_SDK_DIR "${CMAKE_SOURCE_DIR}/vendor/streamline-sdk")
set(STREAMLINE_INCLUDE_DIR "${STREAMLINE_SDK_DIR}/include")
set(STREAMLINE_LIB_DIR "${STREAMLINE_SDK_DIR}/lib/x64")
set(STREAMLINE_BIN_DIR "${STREAMLINE_SDK_DIR}/bin/x64")

# Link library
target_link_libraries(DX12UnitTests PRIVATE
    "${STREAMLINE_LIB_DIR}/sl.interposer.lib"
)

# Copy DLLs to output
add_custom_command(TARGET DX12UnitTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STREAMLINE_BIN_DIR}/sl.interposer.dll"
        $<TARGET_FILE_DIR:DX12UnitTests>
)
```

---

## Next Steps

1. **Implement `StreamlineDLSSWrapper.cpp`** using Streamline API
2. **Update CMakeLists.txt** to link Streamline libraries
3. **Create unit tests** for Streamline initialization
4. **Test on RTX 3070 Ti** with real rendering pipeline
5. **Benchmark performance** across quality modes
6. **Document integration** for future developers

---

## References

- [Streamline Programming Guide](vendor/streamline-sdk/docs/ProgrammingGuide.md)
- [DLSS Programming Guide](vendor/streamline-sdk/docs/ProgrammingGuideDLSS.md)
- [Streamline GitHub](https://github.com/NVIDIA-RTX/Streamline)
- [NVIDIA Developer Portal](https://developer.nvidia.com/rtx/streamline)

---

## Status Summary

✅ **Completed**:
- Streamline SDK 2.9.0 downloaded and vendored
- StreamlineDLSSWrapper header created
- Integration plan documented

🔄 **In Progress**:
- StreamlineDLSSWrapper implementation
- CMake configuration for Streamline

⏳ **Pending**:
- Unit tests for Streamline DLSS
- Performance benchmarking
- Ray tracing integration (next phase)
