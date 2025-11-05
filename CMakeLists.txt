# CMakeLists.txt for AAA Game Engine

cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# Build options
option(BUILD_TESTING "Enable building tests" ON)
option(ENABLE_WARNINGS "Enable extra warnings" ON)
option(ENABLE_CUDA "Enable CUDA features" ON)
option(ENABLE_RENDER_TESTS "Build rendering tests (requires API alignment)" OFF)

# First configure C++ project
project(AAA_Game_Engine LANGUAGES CXX)

# Apply warning settings if enabled (milestone didn't have /WX enabled)
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)  # REMOVED /WX - milestone didn't have it
    else()
        add_compile_options(-Wall -Wextra)
    endif()
endif()

# Enable CUDA language only when requested and available
if(ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    if(CUDAToolkit_ROOT)
        message(STATUS "Using CUDA Toolkit from: ${CUDAToolkit_ROOT}")
    endif()
endif()

# Assets directory for models/textures
set(ASSET_DIR "${CMAKE_SOURCE_DIR}/assets" CACHE PATH "Path to your assets root")

# Standard C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard only when CUDA is enabled
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Detect compiler and set appropriate flags
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "Configuring for MSVC compiler")
    
# Always use release MD runtime to match PhysX libraries
    set(CMAKE_CXX_FLAGS_DEBUG "/MD /O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /DNDEBUG")
    
    # Force NDEBUG globally for PhysX
    add_compile_definitions(NDEBUG)
    
# Use release MD runtime for consistency with PhysX
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    
    # CRITICAL: Ignore static runtime library from PhysX to avoid LNK2038 errors
    # PhysX static libs are built with /MT but we use /MD
    add_link_options(/NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD)

    # Definitions for MSVC builds
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _ITERATOR_DEBUG_LEVEL=0
        GLM_ENABLE_EXPERIMENTAL
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Configuring for Clang compiler")
    # Clang-specific flags
    add_compile_options(
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Release>:-O3>
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -fno-omit-frame-pointer
        $<$<CONFIG:Debug>:-fstandalone-debug>
    )

    if(WIN32)
        # Windows-specific Clang flags
        add_compile_options(
            -D_DEBUG=$<CONFIG:Debug>
            "-fms-compatibility"
            "-fms-extensions"
            "-fdelayed-template-parsing"
        )
    endif()
endif()

# Add NOMINMAX for Windows to prevent min/max macro conflicts
if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# Set CUDA architecture for RTX 3070 Ti
set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA specific flags - only set if CUDA language is enabled
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    
    # Disable CUDA device linking for now
    set(CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Set consistent runtime library - ALWAYS /MD like milestone
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")  # Always /MD, never /MDd

# Ensure consistent runtime for dependencies
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
# NO DEBUG POSTFIX - milestone didn't have it

# Find required packages
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(OpenGL REQUIRED)

# Find GLFW
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
    )
    FetchContent_MakeAvailable(glfw)
endif()

# Find GLAD
find_package(glad QUIET)
if(NOT glad_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v0.1.36
    )
    FetchContent_MakeAvailable(glad)
endif()

# Find GLM - Use FetchContent like milestone
find_package(glm QUIET)
if(NOT glm_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
    )
    FetchContent_MakeAvailable(glm)
endif()

# PhysX Integration (optional)
option(ENABLE_PHYSX "Enable PhysX integration" ON)

# Auto-detect PhysX root. Prefer the 5.6.0 MD build bundled in the repo, falling back to vendor/PhysX/physx.
# Allows manual override via -DPHYSX_ROOT_DIR=...
if(NOT PHYSX_ROOT_DIR)
    set(_PHYSX_CANDIDATE_DIRS
        "${CMAKE_SOURCE_DIR}/vendor/PhysX-107.0-physx-5.6.0/physx"
        "${CMAKE_SOURCE_DIR}/vendor/PhysX/physx"
    )
    foreach(_dir IN LISTS _PHYSX_CANDIDATE_DIRS)
        if(EXISTS "${_dir}/bin/win.x86_64.vc142.md/release/PhysX_64.lib")
            set(PHYSX_ROOT_DIR "${_dir}")
            break()
        endif()
    endforeach()
    if(NOT PHYSX_ROOT_DIR)
        # Default to the 5.6.0 path to match our repository contents
        set(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/vendor/PhysX-107.0-physx-5.6.0/physx")
    endif()
endif()

message(STATUS "PhysX root: ${PHYSX_ROOT_DIR}")
set(PHYSX_INCLUDE_DIR ${PHYSX_ROOT_DIR}/include)

# Auto-detect PhysX bin subdirectory for DLL copy (vc142.md, vc143.md, etc.)
if(MSVC AND ENABLE_PHYSX)
    file(GLOB PHYSX_BIN_CANDIDATES "${PHYSX_ROOT_DIR}/bin/win.x86_64.vc*.md/release")
    list(LENGTH PHYSX_BIN_CANDIDATES CANDIDATE_COUNT)
    if(CANDIDATE_COUNT GREATER 0)
        list(GET PHYSX_BIN_CANDIDATES 0 PHYSX_DLL_DIR)
        message(STATUS "Auto-detected PhysX DLL dir: ${PHYSX_DLL_DIR}")
    else()
        set(PHYSX_DLL_DIR "${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release")
        message(WARNING "PhysX DLL dir not found; using fallback: ${PHYSX_DLL_DIR}")
    endif()
    set(PHYSX_DLL_DIR "${PHYSX_DLL_DIR}" CACHE PATH "PhysX DLL directory")
endif()

if(ENABLE_PHYSX)
    # Configure PhysX paths based on compiler and build type
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(PHYSX_COMPILER_DIR "vc142")
        # Force NDEBUG for PhysX compatibility
        add_compile_definitions(
            NDEBUG
            _ITERATOR_DEBUG_LEVEL=0
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PHYSX_COMPILER_DIR "clang")
    else()
        message(WARNING "Unsupported compiler for PhysX integration; disabling PhysX")
        set(ENABLE_PHYSX OFF)
    endif()
endif()

if(ENABLE_PHYSX)
    # PhysX library suffix selection - NO DEBUG LIBS EXIST
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(PHYSX_LIB_SUFFIX "_64")  # Always use release libs
        set(PHYSX_STATIC_SUFFIX "_static_64")  # Always use release libs
        set(PHYSX_BUILD_TYPE "release")  # Always use release
    else()
        set(PHYSX_LIB_SUFFIX "_static_64")
        set(PHYSX_STATIC_SUFFIX "_static_64")
        set(PHYSX_BUILD_TYPE "$<$<CONFIG:Debug>:debug>$<$<NOT:$<CONFIG:Debug>>:release>")
    endif()

    # Library directory
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PHYSX_LIB_DIR ${PHYSX_ROOT_DIR}/compiler/vc17win64-clang/sdk_source_bin)
        set(PHYSX_LIB_LEVEL "${PHYSX_BUILD_TYPE}")
    else()
        # Use auto-detected PHYSX_DLL_DIR if available, otherwise fallback
        if(DEFINED PHYSX_DLL_DIR)
            set(PHYSX_LIB_DIR ${PHYSX_DLL_DIR})
        else()
            set(PHYSX_LIB_DIR ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release)
        endif()
        set(PHYSX_LIB_LEVEL "")
    endif()

    # Libraries
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PHYSX_LIBRARIES
            ${PHYSX_LIB_DIR}/LowLevel.dir/${PHYSX_LIB_LEVEL}/LowLevel.lib
            ${PHYSX_LIB_DIR}/LowLevelAABB.dir/${PHYSX_LIB_LEVEL}/LowLevelAABB.lib
            ${PHYSX_LIB_DIR}/LowLevelDynamics.dir/${PHYSX_LIB_LEVEL}/LowLevelDynamics.lib
            ${PHYSX_LIB_DIR}/PhysXTask.dir/${PHYSX_LIB_LEVEL}/PhysXTask.lib
            ${PHYSX_LIB_DIR}/SceneQuery.dir/${PHYSX_LIB_LEVEL}/SceneQuery.lib
            ${PHYSX_LIB_DIR}/SimulationController.dir/${PHYSX_LIB_LEVEL}/SimulationController.lib
        )
    else()
        # DLL import libraries + required static libraries for extensions
        set(PHYSX_LIBRARIES
            ${PHYSX_LIB_DIR}/PhysX_64.lib
            ${PHYSX_LIB_DIR}/PhysXCommon_64.lib
            ${PHYSX_LIB_DIR}/PhysXFoundation_64.lib
            ${PHYSX_LIB_DIR}/PhysXCooking_64.lib
            ${PHYSX_LIB_DIR}/PhysXExtensions_static_64.lib
            ${PHYSX_LIB_DIR}/PhysXCharacterKinematic_static_64.lib
            ${PHYSX_LIB_DIR}/PhysXPvdSDK_static_64.lib
            ${PHYSX_LIB_DIR}/SceneQuery_static_64.lib
            ${PHYSX_LIB_DIR}/SimulationController_static_64.lib
        )
    endif()

    # Verify PhysX libraries exist; if missing, disable PhysX to avoid hard errors in CI
    set(_PHYSX_MISSING FALSE)
    foreach(lib ${PHYSX_LIBRARIES})
        if(NOT EXISTS ${lib})
            message(WARNING "PhysX library not found: ${lib}")
            set(_PHYSX_MISSING TRUE)
        endif()
    endforeach()
    if(_PHYSX_MISSING)
        message(WARNING "PhysX libraries missing; disabling PhysX for this build")
        set(ENABLE_PHYSX OFF)
    endif()
endif()

# Configure Assimp
find_package(assimp QUIET)
if(NOT assimp_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG v5.2.5
        CMAKE_ARGS
            -DASSIMP_BUILD_TESTS=OFF
            -DASSIMP_WARNINGS_AS_ERRORS=OFF
            -DBUILD_SHARED_LIBS=OFF
            -DASSIMP_BUILD_STATIC_LIB=ON
            -DASSIMP_FORCE_DEBUG_POSTFIX=OFF
            -DCMAKE_CXX_FLAGS_DEBUG="/MD /O2"
            -DCMAKE_CXX_FLAGS_RELEASE="/MD /O2"
            -DCMAKE_C_FLAGS_DEBUG="/MD /O2"
            -DCMAKE_C_FLAGS_RELEASE="/MD /O2"
            -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL"
    )
    FetchContent_MakeAvailable(assimp)
    set(ASSIMP_INCLUDE_DIR ${assimp_SOURCE_DIR}/include)
    set(ASSIMP_LIBRARY ${assimp_BINARY_DIR}/lib/assimp.lib)
    
    # Make sure assimp uses same runtime library as parent project
    if(TARGET assimp)
        set_target_properties(assimp PROPERTIES
            MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}"
            INTERFACE_MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}"
        )
    endif()

    # Configure runtime for all our targets once we create them
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${PHYSX_INCLUDE_DIR}
)

# Configure all targets to use consistent runtime library
function(configure_msvc_runtime TARGET_NAME)
    if(MSVC)
        set_target_properties(${TARGET_NAME} PROPERTIES
            MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}"
        )
    endif()
endfunction()

# --- Demo Executables ---

# CUDA Physics Demo
add_executable(CudaPhysicsDemo
    src_refactored/Demos/CudaPhysicsDemo.cpp
    src_refactored/Physics/CudaPhysicsSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/Mesh.cpp
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
)

target_include_directories(CudaPhysicsDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${assimp_BINARY_DIR}/include
)

configure_msvc_runtime(CudaPhysicsDemo)

# CUDA Rendering Demo
add_executable(CudaRenderingDemo
    src_refactored/Demos/CudaRenderingDemo.cpp
    src_refactored/Rendering/CudaRenderingSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/Mesh.cpp
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
)

target_include_directories(CudaRenderingDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${assimp_BINARY_DIR}/include
)

configure_msvc_runtime(CudaRenderingDemo)

# Lighting Integration Demo
add_executable(LightingIntegrationDemo
    src_refactored/Demos/LightingIntegrationDemo.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/Mesh.cpp
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
)

target_include_directories(LightingIntegrationDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${assimp_BINARY_DIR}/include
)

configure_msvc_runtime(LightingIntegrationDemo)

# Test Framework and Test Suite
set(TEST_SOURCES
    # Test runner and framework
    tests/TestRunner.cpp
    src_refactored/Testing/TestFramework.cpp
    src_refactored/Testing/TestDebugger.cpp
    
    # Core test suites (always enabled)
    tests/CoreSystemsTests.cpp
    tests/OrbitCameraTests.cpp
    
    # Core dependencies
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
    
    # Rendering dependencies
    src_refactored/Rendering/OrbitCamera.cpp
    src_refactored/Rendering/Camera.cpp
)

# Conditionally add rendering tests
if(ENABLE_RENDER_TESTS)
    list(APPEND TEST_SOURCES tests/RenderingSystemTests.cpp)
endif()

# Conditionally add CUDA-dependent tests (emissive texture generation)
if(ENABLE_CUDA)
    list(APPEND TEST_SOURCES
        tests/EmissiveTextureTests.cpp
        src_refactored/Rendering/CudaBuildingGenerator.cpp
        src_refactored/Rendering/CudaBuildingKernels.cu
    )
endif()

# Conditionally add physics-dependent tests (exclude advanced CUDA perf framework for now)
if(ENABLE_PHYSX)
    list(APPEND TEST_SOURCES 
        tests/CharacterControllerTests.cpp
        # tests/PhysXIntegrationTests.cpp     # temporarily excluded due to AdvancedTestFramework CUDA linkage
        src_refactored/Physics/PhysXPhysicsSystem.cpp
        src_refactored/Physics/WallRunningSystem.cpp
        src_refactored/Gameplay/CharacterControllerSystem.cpp
        # src_refactored/Testing/AdvancedTestFramework.cpp  # excluded: causes cudart unresolveds in TestRunner
    )
endif()

# Note: PlayerMovementTests requires gmock and Full3DGameIntegrationTests needs refactoring
# These are temporarily disabled until dependencies are resolved
# list(APPEND TEST_SOURCES tests/PlayerMovementTests.cpp)
# list(APPEND TEST_SOURCES tests/Full3DGameIntegrationTests.cpp)

add_executable(TestRunner ${TEST_SOURCES})

# Add GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Configure test dependencies
configure_msvc_runtime(TestRunner)

target_include_directories(TestRunner PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${CMAKE_SOURCE_DIR}/src_refactored
    ${CMAKE_SOURCE_DIR}/tests
    $<$<BOOL:${ENABLE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>
)

target_link_libraries(TestRunner PRIVATE
    glm::glm
    GTest::gtest
    GTest::gtest_main
)

if(ENABLE_CUDA)
    # Ensure CUDA kernels compile and link for tests
    set_source_files_properties(
        src_refactored/Rendering/CudaBuildingKernels.cu
        PROPERTIES
        LANGUAGE CUDA
    )
    target_link_libraries(TestRunner PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )
    target_compile_definitions(TestRunner PRIVATE ENABLE_CUDA)
endif()

if(ENABLE_PHYSX)
    target_link_libraries(TestRunner PRIVATE ${PHYSX_LIBRARIES})
    target_compile_definitions(TestRunner PRIVATE ENABLE_PHYSX)
endif()

# Set output directory for test binaries
set_target_properties(TestRunner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
)

# Add test defines
target_compile_definitions(TestRunner PRIVATE
    CUDA_TEST_ENABLED
    GLM_FORCE_CUDA
    _ITERATOR_DEBUG_LEVEL=0
    $<$<BOOL:${ENABLE_RENDER_TESTS}>:ENABLE_RENDER_TESTS>
)

# Configure test output directory
set_target_properties(TestRunner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
    CUDA_SEPARABLE_COMPILATION ON
)

# Register ctest with proper runtime configuration
add_test(NAME TestRunner COMMAND TestRunner)

# Force debug/release defines for TestRunner
target_compile_definitions(TestRunner PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)

# Ensure consistent runtime for TestRunner
set_target_properties(TestRunner PROPERTIES
    MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}"
)

# Copy PhysX DLLs to TestRunner output directory
if(ENABLE_PHYSX AND WIN32)
    if(NOT DEFINED PHYSX_DLL_DIR)
        set(PHYSX_DLL_DIR ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release)
    endif()
    set(PHYSX_DLLS
        ${PHYSX_DLL_DIR}/PhysX_64.dll
        ${PHYSX_DLL_DIR}/PhysXCommon_64.dll
        ${PHYSX_DLL_DIR}/PhysXFoundation_64.dll
        ${PHYSX_DLL_DIR}/PhysXCooking_64.dll
    )
    
    foreach(DLL_FILE ${PHYSX_DLLS})
        add_custom_command(TARGET TestRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:TestRunner>
            COMMENT "Copying PhysX DLL to TestRunner: ${DLL_FILE}"
        )
    endforeach()
endif()

# --- Target Linking ---

# Runtime library already configured above

target_link_libraries(CudaPhysicsDemo PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
)

if(ENABLE_PHYSX)
    target_link_libraries(CudaPhysicsDemo PRIVATE ${PHYSX_LIBRARIES})
    target_compile_definitions(CudaPhysicsDemo PRIVATE ENABLE_PHYSX)
endif()

target_compile_definitions(CudaPhysicsDemo PRIVATE ASSET_DIR="${ASSET_DIR}")

target_link_libraries(CudaRenderingDemo PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
)

target_compile_definitions(CudaRenderingDemo PRIVATE ASSET_DIR="${ASSET_DIR}")

target_link_libraries(LightingIntegrationDemo PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
)

target_compile_definitions(LightingIntegrationDemo PRIVATE ASSET_DIR="${ASSET_DIR}")

target_link_libraries(TestRunner PRIVATE
    glm::glm
)


# --- CUDA Properties ---

set_target_properties(CudaPhysicsDemo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(CudaRenderingDemo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Enhanced Game target with all original features in ECS
add_executable(EnhancedGame
    src_refactored/EnhancedGameMain.cpp
    src_refactored/Gameplay/PlayerMovementSystem.cpp
    src_refactored/Gameplay/EnemyAISystem.cpp
    src_refactored/Gameplay/LevelSystem.cpp
    src_refactored/Gameplay/TargetingSystem.cpp
    # Core ECS
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
    # Physics
    src_refactored/Physics/PhysicsSystem.cpp
    src_refactored/Physics/CudaPhysicsSystem.cpp
    src_refactored/Physics/CollisionDetection.cpp
    # Rendering
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/Mesh.cpp
    # Debug
    src_refactored/Debug/OpenGLDebugRenderer.cpp
    src_refactored/Rendering/RenderDebugSystem.cpp
    # Particles
    src_refactored/Particles/ParticleSystem.cpp
    # Animation
    src_refactored/Animation/AnimationSystem.cpp
)

target_include_directories(EnhancedGame PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${assimp_BINARY_DIR}/include
)

configure_msvc_runtime(EnhancedGame)

target_link_libraries(EnhancedGame PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
)

if(ENABLE_CUDA)
    target_link_libraries(EnhancedGame PRIVATE CUDA::cudart)
    target_compile_definitions(EnhancedGame PRIVATE ENABLE_CUDA)
endif()

if(ENABLE_PHYSX)
    target_link_libraries(EnhancedGame PRIVATE ${PHYSX_LIBRARIES})
    target_compile_definitions(EnhancedGame PRIVATE ENABLE_PHYSX)
endif()

# Expose asset directory to code
target_compile_definitions(EnhancedGame PRIVATE ASSET_DIR="${ASSET_DIR}")

# Copy assets to build directory for runtime
add_custom_command(TARGET EnhancedGame POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:EnhancedGame>/assets
    COMMENT "Copying assets to EnhancedGame build directory"
)

# Full 3D game with all features
add_executable(Full3DGame
    src_refactored/EnhancedGameMain_Full3D.cpp
    src_refactored/Gameplay/PlayerMovementSystem.cpp
    src_refactored/Gameplay/CharacterControllerSystem.cpp
    src_refactored/Gameplay/EnemyAISystem.cpp
    src_refactored/Gameplay/LevelSystem.cpp
    src_refactored/Gameplay/TargetingSystem.cpp
    # Core ECS
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
    # Physics (conditionally include PhysX sources)
    src_refactored/Physics/PhysicsSystem.cpp
    $<$<BOOL:${ENABLE_PHYSX}>:src_refactored/Physics/PhysXPhysicsSystem.cpp>
    src_refactored/Physics/CudaPhysicsSystem.cpp
    src_refactored/Physics/CollisionDetection.cpp
    $<$<BOOL:${ENABLE_PHYSX}>:src_refactored/Physics/WallRunningSystem.cpp>
    # Rendering
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/OrbitCamera.cpp
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/Mesh.cpp
    src_refactored/Rendering/ProceduralCharacter.cpp
    # CUDA Building Generation (conditional - use CUDA or CPU fallback)
    $<$<BOOL:${ENABLE_CUDA}>:src_refactored/Rendering/CudaBuildingGenerator.cpp>
    $<$<BOOL:${ENABLE_CUDA}>:src_refactored/Rendering/CudaBuildingKernels.cu>
    $<$<NOT:$<BOOL:${ENABLE_CUDA}>>:src_refactored/Rendering/CudaBuildingGeneratorFallback.cpp>
    # Debug
    src_refactored/Debug/OpenGLDebugRenderer.cpp
    src_refactored/Rendering/RenderDebugSystem.cpp
    # Particles
    src_refactored/Particles/ParticleSystem.cpp
    # Animation
    src_refactored/Animation/AnimationSystem.cpp
)

target_include_directories(Full3DGame PRIVATE
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${assimp_BINARY_DIR}/include
)

# Configure Full3DGame with MD runtime and correct debug/release definitions
if(MSVC)
    set_target_properties(Full3DGame PROPERTIES
        MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}"
    )
endif()

# CUDA-specific properties for Full3DGame
if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set_target_properties(Full3DGame PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    
    # Set CUDA source properties
    set_source_files_properties(
        src_refactored/Rendering/CudaBuildingKernels.cu
        PROPERTIES
        LANGUAGE CUDA
    )
endif()

target_link_libraries(Full3DGame PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
)

# Add linker flag to ignore PhysX static lib runtime conflicts
if(MSVC)
    target_link_options(Full3DGame PRIVATE /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD)
endif()

if(ENABLE_CUDA)
    target_link_libraries(Full3DGame PRIVATE CUDA::cudart)
    target_compile_definitions(Full3DGame PRIVATE ENABLE_CUDA)
endif()

if(ENABLE_PHYSX)
    target_link_libraries(Full3DGame PRIVATE ${PHYSX_LIBRARIES})
    target_compile_definitions(Full3DGame PRIVATE ENABLE_PHYSX)
endif()

# Expose asset directory to code
target_compile_definitions(Full3DGame PRIVATE ASSET_DIR="${ASSET_DIR}")

# Copy assets to build directory for runtime
add_custom_command(TARGET Full3DGame POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:Full3DGame>/assets
    COMMENT "Copying assets to Full3DGame build directory"
)

# Copy PhysX DLLs to output directory for runtime
if(ENABLE_PHYSX AND WIN32)
    # PHYSX_DLL_DIR is auto-detected above; fallback if not set
    if(NOT DEFINED PHYSX_DLL_DIR)
        set(PHYSX_DLL_DIR ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release)
    endif()
    set(PHYSX_DLLS
        ${PHYSX_DLL_DIR}/PhysX_64.dll
        ${PHYSX_DLL_DIR}/PhysXCommon_64.dll
        ${PHYSX_DLL_DIR}/PhysXFoundation_64.dll
        ${PHYSX_DLL_DIR}/PhysXCooking_64.dll
    )
    
    foreach(DLL_FILE ${PHYSX_DLLS})
        add_custom_command(TARGET Full3DGame POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:Full3DGame>
            COMMENT "Copying PhysX DLL: ${DLL_FILE}"
        )
        add_custom_command(TARGET EnhancedGame POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:EnhancedGame>
            COMMENT "Copying PhysX DLL: ${DLL_FILE}"
        )
        add_custom_command(TARGET CudaPhysicsDemo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:CudaPhysicsDemo>
            COMMENT "Copying PhysX DLL: ${DLL_FILE}"
        )
    endforeach()
endif()
