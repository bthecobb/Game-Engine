# CMakeLists.txt for AAA Game Engine

cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# First configure C++ project
project(AAA_Game_Engine LANGUAGES CXX)

# CUDA toggle (default ON; CI can turn OFF or we install CUDA on runners)
option(ENABLE_CUDA "Enable CUDA features" ON)

# Enable CUDA language only when requested and available
if(ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
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

    # Definitions for MSVC builds
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _ITERATOR_DEBUG_LEVEL=0
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

# Set consistent runtime library and debug flags
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Set debug/release flags properly for PhysX
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-D_DEBUG)
    remove_definitions(-DNDEBUG)
else()
    add_definitions(-DNDEBUG)
    remove_definitions(-D_DEBUG)
endif()

# Ensure consistent runtime for dependencies
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(CMAKE_DEBUG_POSTFIX "d")

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

# GLM Setup - Use submodule instead of FetchContent
set(GLM_DIR "${CMAKE_SOURCE_DIR}/external/glm")
if(NOT EXISTS ${GLM_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "GLM submodule not found. Please run: git submodule update --init --recursive")
endif()

add_subdirectory(${GLM_DIR} ${CMAKE_BINARY_DIR}/glm EXCLUDE_FROM_ALL)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${GLM_DIR})
add_library(glm::glm ALIAS glm)

# PhysX Integration
set(PHYSX_ROOT_DIR ${CMAKE_SOURCE_DIR}/vendor/PhysX/physx)
set(PHYSX_INCLUDE_DIR ${PHYSX_ROOT_DIR}/include)

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
    message(FATAL_ERROR "Unsupported compiler for PhysX integration")
endif()

# Set PhysX library directory based on build type
# Set PhysX library suffixes based on build type (debug/release)
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # For MSVC we can use the generator expression to handle debug/release
    set(PHYSX_LIB_SUFFIX "$<$<CONFIG:Debug>:_64D>$<$<NOT:$<CONFIG:Debug>>:_64>")
    set(PHYSX_STATIC_SUFFIX "$<$<CONFIG:Debug>:_static_64D>$<$<NOT:$<CONFIG:Debug>>:_static_64>")
    set(PHYSX_BUILD_TYPE "$<$<CONFIG:Debug>:debug>$<$<NOT:$<CONFIG:Debug>>:release>")
else()
    # For ClangCL we use static suffixes
    set(PHYSX_LIB_SUFFIX "_static_64")
    set(PHYSX_STATIC_SUFFIX "_static_64")
    set(PHYSX_BUILD_TYPE "$<$<CONFIG:Debug>:debug>$<$<NOT:$<CONFIG:Debug>>:release>")
endif()

# Set PhysX library paths based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For ClangCL we use the source bin directory since we built from source
    set(PHYSX_LIB_DIR ${PHYSX_ROOT_DIR}/compiler/vc17win64-clang/sdk_source_bin)
    set(PHYSX_LIB_LEVEL "${PHYSX_BUILD_TYPE}")
else()
# Use available PhysX MD libs
set(PHYSX_LIB_DIR ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release)  # Only release libs are available
set(PHYSX_LIB_LEVEL "")
endif()

# PhysX Libraries with dynamic debug/release selection
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For ClangCL we use the source bin directory structure
    set(PHYSX_LIBRARIES
        ${PHYSX_LIB_DIR}/LowLevel.dir/${PHYSX_LIB_LEVEL}/LowLevel.lib
        ${PHYSX_LIB_DIR}/LowLevelAABB.dir/${PHYSX_LIB_LEVEL}/LowLevelAABB.lib
        ${PHYSX_LIB_DIR}/LowLevelDynamics.dir/${PHYSX_LIB_LEVEL}/LowLevelDynamics.lib
        ${PHYSX_LIB_DIR}/PhysXTask.dir/${PHYSX_LIB_LEVEL}/PhysXTask.lib
        ${PHYSX_LIB_DIR}/SceneQuery.dir/${PHYSX_LIB_LEVEL}/SceneQuery.lib
        ${PHYSX_LIB_DIR}/SimulationController.dir/${PHYSX_LIB_LEVEL}/SimulationController.lib
    )
else()
# For MSVC we use the pre-built binaries
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

# Verify PhysX libraries exist
foreach(lib ${PHYSX_LIBRARIES})
    if(NOT EXISTS ${lib})
        message(WARNING "PhysX library not found: ${lib}")
    endif()
endforeach()

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
            -DASSIMP_NO_EXPORT=ON
            -DASSIMP_BUILD_ZLIB=ON
            -DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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

# Advanced Test Framework
add_executable(TestRunner
    tests/TestRunner.cpp
    tests/CoreSystemsTests.cpp
    tests/OrbitCameraTests.cpp
    tests/CharacterControllerTests.cpp
    tests/RenderingSystemTests.cpp
    tests/Full3DGameIntegrationTests.cpp
    tests/PhysXIntegrationTests.cpp
    src_refactored/Testing/TestFramework.cpp
    src_refactored/Testing/AdvancedTestFramework.cpp
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
    src_refactored/Physics/PhysXPhysicsSystem.cpp
    src_refactored/Rendering/OrbitCamera.cpp
    src_refactored/Rendering/Camera.cpp
)

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

target_link_libraries(TestRunner PRIVATE
    ${CUDA_LIBRARIES}
    ${PHYSX_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    GTest::gtest
    GTest::gtest_main
    glm::glm
)

# Add test defines
target_compile_definitions(TestRunner PRIVATE
    CUDA_TEST_ENABLED
    GLM_FORCE_CUDA
    _ITERATOR_DEBUG_LEVEL=0
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

# --- Target Linking ---

# Runtime library already configured above

target_link_libraries(CudaPhysicsDemo PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    assimp
    ${PHYSX_LIBRARIES}
)

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
    CUDA::cudart
    assimp
    ${PHYSX_LIBRARIES}
)

# Expose asset directory to code
target_compile_definitions(EnhancedGame PRIVATE ASSET_DIR="${ASSET_DIR}")

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
    # Physics
    src_refactored/Physics/PhysicsSystem.cpp
    src_refactored/Physics/PhysXPhysicsSystem.cpp
    src_refactored/Physics/CudaPhysicsSystem.cpp
    src_refactored/Physics/CollisionDetection.cpp
    src_refactored/Physics/WallRunningSystem.cpp
    # Rendering
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/CameraDebugSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
    src_refactored/Rendering/OrbitCamera.cpp
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

target_link_libraries(Full3DGame PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
    CUDA::cudart
    assimp
    ${PHYSX_LIBRARIES}
)

# Expose asset directory to code
target_compile_definitions(Full3DGame PRIVATE ASSET_DIR="${ASSET_DIR}")
