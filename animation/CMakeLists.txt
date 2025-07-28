cmake_minimum_required(VERSION 3.12)

project(EnhancedAnimationSystem VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W3 /O2 /EHsc)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# Find or download GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found, using bundled version")
    
    # Download GLM if not present
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/glm/glm/glm.hpp")
        message(STATUS "Downloading GLM...")
        include(FetchContent)
        FetchContent_Declare(
            glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
            GIT_TAG        0.9.9.8
        )
        FetchContent_MakeAvailable(glm)
    else()
        # Use local GLM
        add_subdirectory(../third_party/glm ${CMAKE_BINARY_DIR}/glm)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Source files for Animation System
set(ANIMATION_SOURCES
    ../src/AnimationSystem.cpp
    ../src/AnimationUtils.cpp
    ../src/RhythmFeedbackSystem.cpp
)

# Header files
set(ANIMATION_HEADERS
    ../include/AnimationSystem.h
)

# Create Animation System library
add_library(AnimationSystem STATIC ${ANIMATION_SOURCES} ${ANIMATION_HEADERS})

# Link GLM to Animation System
if(TARGET glm::glm)
    target_link_libraries(AnimationSystem PUBLIC glm::glm)
elseif(TARGET glm)
    target_link_libraries(AnimationSystem PUBLIC glm)
else()
    target_include_directories(AnimationSystem PUBLIC ../third_party/glm)
endif()

# Create Simple Animation Demo executable (standalone)
add_executable(SimpleAnimationDemo ../src/SimpleAnimationDemo.cpp)

# Create full Animation Demo executable (with complex dependencies)
# add_executable(AnimationDemo ../src/AnimationDemo.cpp)
# target_link_libraries(AnimationDemo AnimationSystem)

# Platform-specific libraries
if(WIN32)
    # Windows-specific libraries if needed
elseif(UNIX AND NOT APPLE)
    # Linux-specific libraries
    target_link_libraries(AnimationDemo pthread)
elseif(APPLE)
    # macOS-specific libraries if needed
endif()

# Set output directories
set_target_properties(SimpleAnimationDemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(AnimationSystem PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Installation rules
install(TARGETS SimpleAnimationDemo DESTINATION bin)
install(TARGETS AnimationSystem DESTINATION lib)
install(FILES ${ANIMATION_HEADERS} DESTINATION include)

# Testing support
option(BUILD_ANIMATION_TESTS "Build animation tests" OFF)
if(BUILD_ANIMATION_TESTS)
    enable_testing()
    
    # Add Google Test if available
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_executable(AnimationTests
            ../tests/test_animation_system.cpp
            ../tests/test_rhythm_feedback.cpp
        )
        target_link_libraries(AnimationTests AnimationSystem GTest::GTest GTest::Main)
        
        add_test(NAME AnimationSystemTests COMMAND AnimationTests)
    else()
        message(WARNING "Google Test not found, skipping tests")
    endif()
endif()

# Summary
message(STATUS "")
message(STATUS "🎮 Enhanced Animation System Configuration Summary")
message(STATUS "================================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "GLM found: ${glm_FOUND}")
message(STATUS "Build tests: ${BUILD_ANIMATION_TESTS}")
message(STATUS "")
message(STATUS "🎯 Targets to build:")
message(STATUS "  AnimationSystem      - Static library with core animation functionality")
message(STATUS "  SimpleAnimationDemo  - Lightweight animation demonstration executable")
message(STATUS "")
message(STATUS "🚀 To build:")
message(STATUS "  cmake --build . --config ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "🎬 To run demo:")
message(STATUS "  ./bin/SimpleAnimationDemo (Unix)")
message(STATUS "  .\\bin\\SimpleAnimationDemo.exe (Windows)")
message(STATUS "================================================")

# Feature summary
message(STATUS "✨ Features included:")
message(STATUS "  ✅ Complete movement animation cycles (idle, walk, run, sprint)")
message(STATUS "  ✅ Real-time rhythm analysis and beat detection")
message(STATUS "  ✅ Animation synchronization with audio beats")
message(STATUS "  ✅ Procedural animation enhancements")
message(STATUS "  ✅ Event-driven animation system")
message(STATUS "  ✅ Adaptive BPM detection")
message(STATUS "  ✅ Multi-layered feedback system")
message(STATUS "  ✅ Visual and movement modulation")
message(STATUS "  ✅ Cross-platform compatibility")
message(STATUS "")
