# CMakeLists.txt for AAA Game Engine

cmake_minimum_required(VERSION 3.20)
project(AAA_Game_Engine LANGUAGES CXX CUDA)

# Assets directory for models/textures
set(ASSET_DIR "${CMAKE_SOURCE_DIR}/assets" CACHE PATH "Path to your assets root")

# Standard C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add NOMINMAX for Windows to prevent min/max macro conflicts
if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# Use default dynamic runtime (/MD) to match PhysX libraries

# Set CUDA architecture for RTX 3070 Ti
set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Disable CUDA device linking for now
set(CUDA_SEPARABLE_COMPILATION OFF)

# Find required packages
find_package(CUDAToolkit REQUIRED)
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

# Find GLM
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

# PhysX Integration
set(PHYSX_ROOT_DIR ${CMAKE_SOURCE_DIR}/vendor/PhysX/physx)
set(PHYSX_INCLUDE_DIR ${PHYSX_ROOT_DIR}/include)
set(PHYSX_LIB_DIR ${PHYSX_ROOT_DIR}/bin/win.x86_64.vc142.md/release)

# PhysX Libraries
set(PHYSX_LIBRARIES
    ${PHYSX_LIB_DIR}/PhysX_64.lib
    ${PHYSX_LIB_DIR}/PhysXCommon_64.lib
    ${PHYSX_LIB_DIR}/PhysXFoundation_64.lib
    ${PHYSX_LIB_DIR}/PhysXExtensions_static_64.lib
    ${PHYSX_LIB_DIR}/PhysXCooking_64.lib
    ${PHYSX_LIB_DIR}/PhysXCharacterKinematic_static_64.lib
    ${PHYSX_LIB_DIR}/PhysXVehicle2_static_64.lib
    ${PHYSX_LIB_DIR}/PhysXPvdSDK_static_64.lib
    ${PHYSX_LIB_DIR}/SceneQuery_static_64.lib
    ${PHYSX_LIB_DIR}/SimulationController_static_64.lib
)

find_package(assimp QUIET)
if(NOT assimp_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG v5.2.5
        CMAKE_ARGS     -DASSIMP_BUILD_TESTS=OFF
    )
    FetchContent_MakeAvailable(assimp)
    set(ASSIMP_INCLUDE_DIR ${assimp_SOURCE_DIR}/include)
    set(ASSIMP_LIBRARY ${assimp_BINARY_DIR}/lib/assimp.lib)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include_refactored
    ${ASSIMP_INCLUDE_DIR}
    ${PHYSX_INCLUDE_DIR}
)

# --- Demo Executables ---

# CUDA Physics Demo
add_executable(CudaPhysicsDemo
    src_refactored/Demos/CudaPhysicsDemo.cpp
    src_refactored/Physics/CudaPhysicsSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
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

# CUDA Rendering Demo
add_executable(CudaRenderingDemo
    src_refactored/Demos/CudaRenderingDemo.cpp
    src_refactored/Rendering/CudaRenderingSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
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

# Lighting Integration Demo
add_executable(LightingIntegrationDemo
    src_refactored/Demos/LightingIntegrationDemo.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/RenderSystem.cpp
    src_refactored/Rendering/ShaderProgram.cpp
    src_refactored/Rendering/Camera.cpp
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

# Test Runner
add_executable(TestRunner
    tests/TestRunner.cpp
    tests/CoreSystemsTests.cpp
    src_refactored/Testing/TestFramework.cpp
    src_refactored/Core/EntityManager.cpp
    src_refactored/Core/ComponentManager.cpp
)

# --- Target Linking ---

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
    src_refactored/Rendering/Framebuffer.cpp
    src_refactored/Rendering/LightingSystem.cpp
    src_refactored/Rendering/Mesh.cpp
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
