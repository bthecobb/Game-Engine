cmake_minimum_required(VERSION 3.20)
project(CudaGame LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
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

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Include directories
include_directories(include)

# Create only RhythmArenaDemo executable to avoid compilation errors
add_executable(RhythmArenaDemo
    src/animation/RhythmArenaDemo.cpp
    src/animation/SimpleAnimationController.cpp
    src/CharacterMesh.cpp
)

# Link libraries for RhythmArenaDemo
target_link_libraries(RhythmArenaDemo 
    ${OPENGL_LIBRARIES}
    glfw
    glad
    glm::glm
)

# Set include directories for RhythmArenaDemo
target_include_directories(RhythmArenaDemo PRIVATE src/animation)
